//! `HeldTablet` — summoned HUD tile for the config-panel flow.
//!
//! When the player presses F on a functional block, the client spawns
//! a `HeldTablet` tile positioned ~60 cm in front of the camera at a
//! comfortable reading angle, with a `ConfigPanelWidget` populated
//! from the server-sent `BlockSignalConfig`. The tile despawns on
//! Apply / Close / Esc / F.
//!
//! Today the tablet is **camera-relative** — a transform follower
//! keeps it at a fixed offset from the player's eye. Future: when an
//! avatar rig lands, move the follower's anchor to the
//! `avatar-left-hand` bone — zero architecture change.

use bevy::prelude::*;

use voxeldust_core::signal::config::BlockSignalConfig;

use crate::hud::material::new_egui_target_image;
use crate::hud::tablet_ui::spawn_tablet_egui_camera;
use crate::hud::tile::{
    HudAttachment, HudConfig, HudPayload, HudTexture, HudTile, WidgetKind,
};
use crate::shard::ShardKey;

/// Marker component on the one (or zero) currently-held tablet.
#[derive(Component, Debug, Clone, Copy)]
pub struct HeldTablet;

/// Spawn-in animation state. Elapsed seconds since spawn; the
/// `animate_tablet_spawn` system scales the tablet from
/// `SPAWN_SCALE_START` → `1.0` over `SPAWN_DURATION`, with a
/// slight y-translation lift. Despawns `TabletSpawnAnim` once
/// the animation completes so the scale settles at 1.
#[derive(Component, Debug, Clone, Copy)]
pub struct TabletSpawnAnim {
    pub elapsed: f32,
}

const SPAWN_DURATION: f32 = 0.22;
const SPAWN_SCALE_START: f32 = 0.25;

/// Fire to summon a tablet with the given `BlockSignalConfig`.
#[derive(Message, Debug, Clone)]
pub struct SpawnHeldTablet {
    pub shard: ShardKey,
    pub config: BlockSignalConfig,
}

/// Fire to despawn the currently-held tablet (Close / Apply / Esc /
/// F-toggle).
#[derive(Message, Debug, Clone, Copy)]
pub struct DespawnHeldTablet;

pub struct HeldTabletPlugin;

impl Plugin for HeldTabletPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<SpawnHeldTablet>()
            .add_message::<DespawnHeldTablet>()
            .add_systems(
                Update,
                (
                    spawn_tablet,
                    despawn_tablet,
                    follow_camera,
                    animate_tablet_spawn,
                ),
            );
    }
}

/// Width / height in world meters. Square 1:1 tablet matches the
/// 512×512 texture — no aspect stretching, content fills the full
/// face. 0.30 m is small enough that at the 0.52 m offset below
/// both edges sit safely inside the default 60° vertical FOV even
/// when tilted at reading angle.
const TABLET_W_M: f32 = 0.30;
const TABLET_H_M: f32 = 0.30;
/// Pixel resolution per tile. Square to match the mesh.
const TABLET_RES: u32 = 512;
/// Reading-angle tilt — rotates the tablet around its local X-axis so
/// the top edge tips away from the pilot, matching a "held" reading
/// pose. 12° is enough to feel held without shortening the visible
/// height too much.
const TABLET_READING_TILT_RAD: f32 = -0.21;
/// Redraw floor cadence — widgets don't need to redraw every frame.
const TABLET_REDRAW_FLOOR_MS: u64 = 50;

/// Offset from the camera in camera-local coordinates. Centered
/// horizontally, slightly below eye-line, arm's-length forward. At
/// these values the tablet subtends ≈ 14° below the camera's forward
/// direction, well inside Bevy's default vertical half-FOV (~22°),
/// so it stays in-frame regardless of the ship's orientation. The
/// earlier offset of `(0.12, -0.16, -0.32)` placed the tablet at
/// ~27° below center — outside the frustum when the ship was tilted
/// nose-down.
/// Camera-local offset. Closer than the original (0.52 m vs 0.58 m)
/// so the square tablet subtends a larger vertical angle — more
/// readable. `y = -0.14` keeps the centre slightly below eye-line
/// (~14° down at this distance); combined with the 0.30 m height and
/// 12° tilt, the tablet's bottom edge lands ~22° below horizontal —
/// well inside the 30° vertical half-FOV Bevy's Camera3d defaults to.
const TABLET_OFFSET_LOCAL: Vec3 = Vec3::new(0.0, -0.14, -0.52);

fn spawn_tablet(
    mut events: MessageReader<SpawnHeldTablet>,
    existing: Query<Entity, With<HeldTablet>>,
    camera: Query<&GlobalTransform, With<crate::MainCamera>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    for ev in events.read() {
        // Replace any existing tablet.
        for e in existing.iter() {
            commands.entity(e).despawn();
        }
        // Tablet image doubles as an egui render target; use the
        // variant that sets `RENDER_ATTACHMENT | COPY_DST |
        // TEXTURE_BINDING` so bevy's render graph can bind it as a
        // colour target for `tablet_ui::TabletPaintPass`.
        let image = new_egui_target_image(&mut images, TABLET_RES);
        // Production tablet material:
        //   * base_color WHITE + texture → widget's RGB pass-through.
        //   * unlit so widget colors aren't darkened by ship ambient.
        //   * double-sided + no-cull so the tablet reads from both
        //     sides (useful when the player rotates past 90°).
        let mat = StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(image.clone()),
            emissive: LinearRgba::new(0.0, 0.0, 0.0, 0.0),
            alpha_mode: AlphaMode::Opaque,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        };
        let material = materials.add(mat);
        let mesh = meshes.add(Rectangle::new(TABLET_W_M, TABLET_H_M).mesh());

        // Compute a sane initial Transform from the camera state we
        // can see THIS FRAME, so the tablet is visible from the very
        // first rendered frame after spawn — no waiting for the
        // `follow_camera` system on the next tick.
        let (cam_pos, cam_rot) = camera
            .single()
            .map(|gt| (gt.translation(), gt.rotation()))
            .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
        // Face the camera directly. Rectangle mesh normal is local +Z;
        // applying `cam_rot` maps that to the world direction opposite
        // the camera's forward (+Z-local = BACK of the camera), which
        // is the direction from the tablet toward the camera. Front
        // faces the eye. No extra pitch — pitch compounds with the
        // ship's nose-down orientation and kicks the tablet out of
        // frame. If we want a holding-angle look, we'll add it back
        // as a small, well-bounded rotation around tile-local X only
        // after the tablet reliably appears.
        // Face the camera + tilt back so the tablet reads as "held"
        // rather than floating flat. Tilt is applied in the TABLET's
        // local X axis (after the face-camera rotation), so it's
        // orientation-independent: regardless of which way the ship
        // is pointing, the tablet's TOP edge always tips away from
        // the pilot.
        let initial_rotation = cam_rot * Quat::from_rotation_x(TABLET_READING_TILT_RAD);
        let initial_translation = cam_pos + cam_rot * TABLET_OFFSET_LOCAL;

        // Diagnostic cube removed — tablet visibility confirmed. If
        // further diagnostics are needed, spawn a separate marker
        // component (NOT `HeldTablet`, otherwise `follow_camera`
        // co-positions it with the tablet and they overlap).

        let tablet_entity_id = commands.spawn((
            HeldTablet,
            TabletSpawnAnim { elapsed: 0.0 },
            HudTile {
                attachment: HudAttachment::Tablet,
            },
            HudConfig {
                kind: WidgetKind::ConfigPanel,
                channel: String::new(),
                property: voxeldust_core::signal::types::SignalProperty::Throttle,
                caption: format!(
                    "Block @ ({}, {}, {})",
                    ev.config.block_pos.x, ev.config.block_pos.y, ev.config.block_pos.z
                ),
                opacity: 0.92,
                // AR ON for the held tablet so you see celestial-body
                // markers projected through its plane — proves the AR
                // pipeline end-to-end against data the client already
                // has (`PrimaryWorldState.bodies[]` +
                // `SecondaryWorldStates`). Wall-mounted tiles get a
                // config-panel toggle for this; the tablet defaults on.
                ar_enabled: true,
                ar_filter: crate::hud::ar::ArFilter {
                    celestial_bodies: true,
                    remote_ships: true,
                    remote_players: true,
                    debris: true,
                },
                payload: HudPayload::ConfigPanel(Box::new(ev.config.clone())),
            },
            HudTexture {
                handle: image.clone(),
                material: material.clone(),
                size: TABLET_RES,
                world_size: Vec2::new(TABLET_W_M, TABLET_H_M),
                last_draw_tick: 0,
                redraw_floor_ms: TABLET_REDRAW_FLOOR_MS,
                // Back-date so `redraw_hud_textures` draws on the
                // very next frame — no invisible-for-50 ms flash
                // between spawn and first paint.
                last_draw_at: std::time::Instant::now()
                    - std::time::Duration::from_millis(TABLET_REDRAW_FLOOR_MS + 1),
            },
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform {
                translation: initial_translation,
                rotation: initial_rotation,
                scale: Vec3::ONE,
            },
            GlobalTransform::IDENTITY,
            // `Visibility::Visible` instead of default (Inherited) so
            // the tablet shows without needing a parent's visibility
            // to resolve.
            Visibility::Visible,
            InheritedVisibility::default(),
            ViewVisibility::default(),
            Name::new(format!("held_tablet[{}]", ev.shard)),
        )).id();

        // Spawn the egui render-to-image camera targeting this
        // tablet's Image. It's a child of the tablet so despawning
        // the tablet despawns the camera + its egui context atomically.
        spawn_tablet_egui_camera(&mut commands, image, tablet_entity_id);
        let _ = tablet_entity_id;
        tracing::info!(
            shard = %ev.shard,
            block = ?(ev.config.block_pos.x, ev.config.block_pos.y, ev.config.block_pos.z),
            cam_pos = ?(cam_pos.x, cam_pos.y, cam_pos.z),
            cam_fwd = ?(cam_rot * Vec3::NEG_Z),
            "held tablet spawned",
        );
    }
}

fn despawn_tablet(
    mut events: MessageReader<DespawnHeldTablet>,
    existing: Query<Entity, With<HeldTablet>>,
    mut commands: Commands,
) {
    for _ in events.read() {
        for e in existing.iter() {
            commands.entity(e).despawn();
            tracing::info!("held tablet despawned");
        }
    }
}

/// Keep the tablet pinned at a fixed offset in camera-local space
/// every frame. Camera-relative rather than parented-to-camera so the
/// render transform can still be rebased by the floating-origin
/// system without fighting the tablet's own Transform.
fn follow_camera(
    cam: Query<&GlobalTransform, With<crate::MainCamera>>,
    mut tablet: Query<(&mut Transform, Option<&TabletSpawnAnim>), With<HeldTablet>>,
) {
    let Ok(cam_gt) = cam.single() else { return };
    let Ok((mut tf, anim)) = tablet.single_mut() else { return };
    let cam_translation: Vec3 = cam_gt.translation();
    let cam_rot = cam_gt.rotation();
    let local_offset = cam_rot * TABLET_OFFSET_LOCAL;
    tf.translation = cam_translation + local_offset;
    // Face the camera + reading-angle tilt (see constant above).
    // Front normal = cam_rot * +Z = direction back toward the
    // player's eye; the extra local-X rotation tips the top edge
    // away so it feels held rather than floating.
    tf.rotation = cam_rot * Quat::from_rotation_x(TABLET_READING_TILT_RAD);

    // Hold the scale from the animation system. When there's no
    // `TabletSpawnAnim`, the animation is complete — scale is 1.
    // The scale is written here (not in `animate_tablet_spawn`)
    // because `follow_camera` is the sole writer of `Transform` for
    // the tablet; writing from both causes one-frame flicker.
    let scale = anim
        .map(|a| spawn_scale_from_elapsed(a.elapsed))
        .unwrap_or(1.0);
    tf.scale = Vec3::splat(scale);
}

/// Tick the spawn animation elapsed timer. Runs ahead of
/// `follow_camera` so the scale this frame reflects this frame's
/// elapsed. When the animation completes, the component is removed,
/// leaving the tablet at scale 1.
fn animate_tablet_spawn(
    time: Res<Time>,
    mut tablet: Query<(Entity, &mut TabletSpawnAnim)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    for (entity, mut anim) in &mut tablet {
        anim.elapsed += dt;
        if anim.elapsed >= SPAWN_DURATION {
            commands.entity(entity).remove::<TabletSpawnAnim>();
        }
    }
}

/// Ease-out cubic scale curve: starts at `SPAWN_SCALE_START`, reaches
/// 1 at `SPAWN_DURATION`. Ease-out = `1 - (1 - t)^3` so the tablet
/// pops in quickly at first, settles smoothly at full size.
fn spawn_scale_from_elapsed(elapsed: f32) -> f32 {
    let t = (elapsed / SPAWN_DURATION).clamp(0.0, 1.0);
    let eased = 1.0 - (1.0 - t).powi(3);
    SPAWN_SCALE_START + (1.0 - SPAWN_SCALE_START) * eased
}
