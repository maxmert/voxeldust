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

use voxeldust_core::client_message::ClientMsg;
use voxeldust_core::signal::config::BlockSignalConfig;
use voxeldust_core::wire_codec;

use crate::character::{BoneRegistry, CharacterAssetRegistry, LocalCharacterTag, RemoteCharacterTag};
use crate::hud::focus::{HudFocusOrigin, HudFocusState};
use crate::hud::material::new_egui_target_image;
use crate::hud::tablet_ui::spawn_tablet_egui_camera;
use crate::hud::tile::{
    HudAttachment, HudConfig, HudPayload, HudTexture, HudTile, WidgetKind,
};
use crate::net::TcpSender;
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

/// System set the despawn handler lives in. Exported so the
/// auto-save flush in `config_panel::save_on_tablet_despawn` can
/// order itself BEFORE the entity goes away — without this label,
/// Bevy's parallel scheduler is free to despawn the tablet first,
/// and the flush would miss the still-populated editable buffer.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TabletDespawnSet;

pub struct HeldTabletPlugin;

impl Plugin for HeldTabletPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<SpawnHeldTablet>()
            .add_message::<DespawnHeldTablet>()
            .init_resource::<TabletCursorStreamState>()
            .add_systems(
                Update,
                (
                    spawn_tablet,
                    despawn_tablet.in_set(TabletDespawnSet),
                    follow_camera,
                    animate_tablet_spawn,
                    stream_tablet_cursor,
                ),
            );
    }
}

/// Phase J — sender state for the throttled cursor stream. The
/// last-sent UV is kept so we suppress redundant ticks (player
/// holding the cursor still). Last-sent time is kept so we cap the
/// stream at ~20 Hz; well-below the default 60 Hz client loop and
/// well-above the 20 Hz server tick where cursor data ultimately
/// matters for remote replication.
#[derive(Resource, Debug, Clone, Copy)]
struct TabletCursorStreamState {
    last_uv: Vec2,
    last_send: f64,
}

impl Default for TabletCursorStreamState {
    fn default() -> Self {
        Self {
            // (-1, -1) is unreachable — `cursor_uv` is always clamped
            // to `[0, 1]` — so the first tick after the tablet opens
            // always passes the "value changed" gate even if the
            // cursor sits dead-centre.
            last_uv: Vec2::new(-1.0, -1.0),
            last_send: f64::NEG_INFINITY,
        }
    }
}

/// Phase J — at ~20 Hz, push `HudFocusState.cursor_uv` to the server
/// so remote viewers' finger-IK tracks the local player's cursor.
/// No-op when no tablet is held or focus is not on the tablet.
/// Suppresses redundant sends by gating on a 1-pixel-equivalent UV
/// delta (≈ 0.0025) — the same threshold the cursor accumulator
/// uses, so any meaningful movement always replicates.
fn stream_tablet_cursor(
    focus: Res<HudFocusState>,
    tablet: Query<(), With<HeldTablet>>,
    tcp: Res<TcpSender>,
    time: Res<Time>,
    mut state: ResMut<TabletCursorStreamState>,
) {
    /// 50 ms = 20 Hz. Matches the server-shard tick so each broadcast
    /// carries the freshest cursor sample (within one client frame
    /// of the broadcast).
    const MIN_INTERVAL_S: f64 = 0.05;
    /// 1-pixel delta in UV space at the focus cursor's sensitivity.
    /// Below this, treat as "no movement" and skip the wire write.
    const MIN_DELTA: f32 = 0.0025;

    if tablet.is_empty() || !matches!(focus.origin, HudFocusOrigin::Tablet) {
        // Tablet closed or focus moved — reset so the next open
        // re-syncs from a clean slate.
        if state.last_uv != Vec2::new(-1.0, -1.0) {
            *state = TabletCursorStreamState::default();
        }
        return;
    }
    let now = time.elapsed().as_secs_f64();
    if now - state.last_send < MIN_INTERVAL_S {
        return;
    }
    let delta = (focus.cursor_uv - state.last_uv).abs();
    if delta.x < MIN_DELTA && delta.y < MIN_DELTA && state.last_uv.x >= 0.0 {
        return;
    }
    // bevy's `Vec2` is `glam` 0.30; `ClientMsg` uses workspace-pinned
    // `glam` 0.29. Bridge by component-wise construction.
    let msg = ClientMsg::TabletCursorUpdate(glam::Vec2::new(focus.cursor_uv.x, focus.cursor_uv.y));
    let bytes = msg.serialize();
    let mut pkt = Vec::new();
    wire_codec::encode(&bytes, &mut pkt);
    if tcp.tx.send(pkt).is_err() {
        tracing::warn!("TCP channel closed while sending TabletCursorUpdate");
        return;
    }
    state.last_uv = focus.cursor_uv;
    state.last_send = now;
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
                // Tablet is always single-widget (a `ConfigPanel`).
                layout: crate::hud::tile::HudPanelLayout::Single,
                extra_slots: None,
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

/// Phase J — AAA "in-hand" architecture. The tablet entity is reparented
/// to the local player's RIGHT HAND bone the first frame both bones
/// resolve. From that point on Bevy's transform propagation drives the
/// tablet's world pose entirely from the hand bone's pose — body sway,
/// walk bobs, IK adjustments, every animation frame propagates 1:1
/// without a per-frame world-write race.
///
/// To make the constant local Transform deterministic, `tablet_ik`
/// **overrides the right wrist's world rotation** to a fixed grip pose
/// (`visual_rot * class.right_hand_grip_rotation_in_body`). With the
/// wrist locked to body-aligned axes, the tablet's local Transform is
/// a closed-form expression of the desired body-local pose:
///
///   * Wrist sits at the tablet's right edge (W/2 to body-LEFT of the
///     screen centre), pulled slightly back along body forward so the
///     hand wraps the back of the bezel.
///   * Mesh axes map to body axes: mesh +X → body right, mesh +Y →
///     body up (pitched toward the face), mesh +Z → body back (pitched
///     toward the face — i.e. the screen normal points at the player).
///
/// Fallback (`cam`): camera-relative pinning until the visual + right
/// hand bone resolve, then hand-parented every frame after.
fn follow_camera(
    mut commands: Commands,
    cam: Query<&GlobalTransform, With<crate::MainCamera>>,
    local: Query<(&BoneRegistry, &RemoteCharacterTag), With<LocalCharacterTag>>,
    asset_registry: Res<CharacterAssetRegistry>,
    mut tablet: Query<
        (Entity, &mut Transform, Option<&ChildOf>, Option<&TabletSpawnAnim>),
        With<HeldTablet>,
    >,
) {
    let Ok((tablet_e, mut tf, child_of, anim)) = tablet.single_mut() else { return };
    let scale = anim
        .map(|a| spawn_scale_from_elapsed(a.elapsed))
        .unwrap_or(1.0);

    let attached = local.single().ok().and_then(|(bones, tag)| {
        if !bones.resolved {
            return None;
        }
        let assets = asset_registry.ready(tag.class_id)?;
        let hand_e = bones.get(assets.class.right_hand_bone)?;
        Some((hand_e, assets.class))
    });

    if let Some((hand_e, class)) = attached {
        let already_parented = child_of.map(|c| c.parent() == hand_e).unwrap_or(false);
        if !already_parented {
            // The naive `commands.entity(hand_e).add_child(tablet_e)`
            // panics in `apply_deferred` if EITHER entity is despawned
            // by the time the buffer applies. The race we hit:
            //   * Tick T: spawn_tablet queues despawn(old_tablet) +
            //     spawn(new_tablet). follow_camera also runs in T,
            //     reads `tablet.single_mut()` which still sees the
            //     OLD tablet (despawn isn't applied yet), queues
            //     `add_child(hand_e, old_tablet)`.
            //   * apply_deferred (FIFO): old_tablet despawn applies,
            //     then add_child runs against the dead entity →
            //     panic in the relationship hook at
            //     `bevy_ecs/relationship/related_methods.rs:46`.
            //
            // Bevy's `commands.get_entity` checks the LIVE entity
            // table at queue time, but a same-tick queued despawn
            // hasn't applied yet — so it returns Ok and we still
            // crash at apply time.
            //
            // Fix: queue a world-level closure that re-validates
            // BOTH entities at apply time, after all earlier-queued
            // commands have run. If either is dead, silently skip;
            // the next frame will re-attempt with fresh entities.
            let parent = hand_e;
            let child = tablet_e;
            commands.queue(move |world: &mut bevy::ecs::world::World| {
                if world.get_entity(parent).is_ok() && world.get_entity(child).is_ok() {
                    world.entity_mut(parent).add_child(child);
                }
            });
        }

        // Closed-form local Transform derivation. Symbols:
        //   * `grip` = class.right_hand_grip_rotation_in_body, which
        //     `tablet_ik` enforces as the wrist's rotation in body-local.
        //   * `inv = grip⁻¹`. Maps body-local axes BACK into bone-local
        //     axes (the frame the tablet's local Transform lives in).
        //
        // Body axes in body-local: body_left = +X, body_up = +Y,
        // body_back = -Z, body_right = -X, body_forward = +Z.
        //
        // Translation: tablet centre = chest + visual_rot * hold_offset.
        // Wrist (post-IK) = chest + visual_rot * (hold_offset.x - W/2,
        //                              hold_offset.y, hold_offset.z + grip_back).
        // Tablet relative to wrist in body-local =
        //   (W/2, 0, -grip_back) — W/2 to body-LEFT, grip_back behind
        //   the screen plane (toward body forward).
        // In hand-local: inv · (W/2, 0, -grip_back).
        //
        // Rotation: each mesh axis maps to a body axis (pitched).
        //   mesh +X → body_right = -X body-local. In hand-local: inv · -X.
        //   mesh +Y → pitched body_up. Pitch axis = body_right (-X body).
        //   mesh +Z → pitched body_back = pitched -Z body.
        // In hand-local, pitch axis = inv · -X.
        let grip = crate::character::tablet_ik::right_hand_grip_in_body_rot(class);
        let inv = grip.inverse();

        let half_w = class.tablet_width / 2.0;
        // Tablet centre relative to the wrist in body-local frame.
        // Mirror of `tablet_ik`'s right_grip_world derivation: wrist
        // sits at `(-half_w - outside_offset, 0, +grip_back_depth)`
        // from the pad centre, so pad centre is the negation:
        // `(+half_w + outside_offset, 0, -grip_back_depth)`. Both
        // must use the SAME outside_offset or wrist and pad drift
        // apart visually.
        let body_local_offset = Vec3::new(
            half_w + class.tablet_right_hand_outside_offset,
            0.0,
            -class.tablet_grip_back_depth,
        );
        let tablet_local_pos = inv * body_local_offset;

        let pitch_axis_local = inv * Vec3::NEG_X;
        let pitch = Quat::from_axis_angle(pitch_axis_local, class.tablet_hold_pitch);
        let basis = Mat3::from_cols(
            inv * Vec3::NEG_X,            // mesh +X → body right
            pitch * (inv * Vec3::Y),      // mesh +Y → body up (pitched)
            pitch * (inv * Vec3::NEG_Z),  // mesh +Z → body back (pitched, screen → player)
        );

        tf.translation = tablet_local_pos;
        tf.rotation = Quat::from_mat3(&basis);
        tf.scale = Vec3::splat(scale);
        return;
    }

    // Fallback before the local visual's bones resolve. Once the
    // tablet is parented this branch never runs again.
    let Ok(cam_gt) = cam.single() else { return };
    let cam_translation: Vec3 = cam_gt.translation();
    let cam_rot = cam_gt.rotation();
    let local_offset = cam_rot * TABLET_OFFSET_LOCAL;
    tf.translation = cam_translation + local_offset;
    tf.rotation = cam_rot * Quat::from_rotation_x(TABLET_READING_TILT_RAD);
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
