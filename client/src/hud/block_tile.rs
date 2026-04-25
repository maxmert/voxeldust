//! Spawn `HudTile` entities for `SubBlockType::HudPanel` elements
//! found in a chunk. Called from the chunk-stream remesh path so that
//! every snapshot / delta re-derives the set of HUD tiles this chunk
//! exposes.
//!
//! One tile = one `(Mesh3d(Rectangle), MeshMaterial3d(StandardMaterial),
//! HudTile, HudConfig, HudTexture)` child entity on the chunk. The
//! tile's local `Transform` orients the plane to match the hosting
//! block's face, sitting flush against the face with its +Z normal
//! pointing outward.
//!
//! Today the `HudConfig` is a client-side default (Gauge widget on an
//! empty channel). Once the server protocol grows a
//! `SubBlockConfigState { block_pos, face, widget_kind, channel,
//! property, ar_flags, … }` message, this module's defaults are
//! replaced by server-authored per-panel config — the spawn wiring
//! doesn't change.

use bevy::prelude::*;

use voxeldust_core::block::{
    chunk_storage::ChunkStorage,
    palette::{index_to_xyz, CHUNK_SIZE},
    sub_block::SubBlockType,
};

use crate::hud::ar::ArFilter;
use crate::hud::material::new_transparent_hud_image;
use crate::hud::panel_config::{HudPanelConfigs, HudPanelKey};
use crate::hud::tile::{HudAttachment, HudConfig, HudPayload, HudTexture, HudTile, WidgetKind};
use crate::shard::ShardKey;

/// Default pixel resolution for block-face HUD panels. Matches the
/// held-tablet default for visual consistency across sources.
const BLOCK_TILE_RES: u32 = 256;

/// Redraw floor cadence for block-face HUD panels. 50 ms covers 20 Hz
/// signal updates + cursor / AR smoothness without burning CPU every
/// frame.
const BLOCK_TILE_REDRAW_FLOOR_MS: u64 = 50;

/// World-space size of a HUD panel in block units. 0.9 × 0.9 leaves a
/// 0.05 bezel on each edge so the panel visibly sits on the face
/// without z-fighting with the host block's geometry.
const PANEL_SIZE_BLOCKS: f32 = 0.9;

/// Thickness-offset from the block face so the panel doesn't z-fight.
/// Tiny (millimetric at block = 1 m scale) but enough for depth
/// testing.
const PANEL_FACE_OFFSET: f32 = 0.01;

/// Spawn one HudTile child under `chunk_entity` for every HudPanel
/// sub-block in `chunk`. Positions are chunk-local; the chunk's own
/// Transform anchors it into the shard.
pub fn spawn_hud_tiles_for_chunk(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    panel_configs: &HudPanelConfigs,
    chunk_entity: Entity,
    chunk: &ChunkStorage,
    shard: ShardKey,
    chunk_index: IVec3,
) {
    for (flat_idx, elements) in chunk.iter_sub_blocks() {
        let (bx, by, bz) = index_to_xyz(flat_idx as usize);
        for elem in elements {
            if elem.element_type != SubBlockType::HudPanel {
                continue;
            }
            let world_block =
                chunk_index * CHUNK_SIZE as i32 + IVec3::new(bx as i32, by as i32, bz as i32);
            let key = HudPanelKey {
                shard,
                block_pos: world_block,
                face: elem.face,
            };
            let settings = panel_configs.get_or_default(key);
            spawn_tile(
                commands,
                meshes,
                materials,
                images,
                chunk_entity,
                shard,
                chunk_index,
                IVec3::new(bx as i32, by as i32, bz as i32),
                elem.face,
                settings,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_tile(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    chunk_entity: Entity,
    shard: ShardKey,
    chunk_index: IVec3,
    block_pos_local: IVec3,
    face: u8,
    settings: crate::hud::panel_config::HudPanelSettings,
) {
    // Transparent background: widgets draw opaque pixels where they
    // have content; empty areas stay alpha=0 so the player sees the
    // block underneath through the HUD panel.
    let image = new_transparent_hud_image(images, BLOCK_TILE_RES);
    let mat = StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(image.clone()),
        emissive: LinearRgba::new(0.0, 0.0, 0.0, 0.0),
        // `AlphaMode::Blend` so alpha=0 pixels are fully see-through.
        // `Mask` would snap to opaque/fully-clear; we want smooth
        // widget anti-aliasing so Blend is correct.
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    };
    let material = materials.add(mat);

    let mesh = meshes.add(
        Rectangle::new(PANEL_SIZE_BLOCKS, PANEL_SIZE_BLOCKS).mesh(),
    );

    let transform = panel_face_transform(block_pos_local, face);

    // Full block position = chunk_index * CHUNK_SIZE + block_pos_local.
    let world_block = chunk_index * CHUNK_SIZE as i32 + block_pos_local;

    commands.spawn((
        HudTile {
            attachment: HudAttachment::Block {
                shard,
                block_pos: world_block,
                face,
            },
        },
        HudConfig {
            kind: settings.slots[0].kind,
            channel: settings.slots[0].channel.clone(),
            property: settings.slots[0].property,
            caption: settings.slots[0].caption.clone(),
            opacity: settings.opacity,
            ar_enabled: settings.ar_enabled,
            ar_filter: settings.ar_filter,
            payload: HudPayload::None,
            layout: settings.layout,
            extra_slots: match settings.layout {
                crate::hud::tile::HudPanelLayout::Single => None,
                crate::hud::tile::HudPanelLayout::Quad => Some(Box::new([
                    settings.slots[1].clone(),
                    settings.slots[2].clone(),
                    settings.slots[3].clone(),
                ])),
            },
        },
        HudTexture {
            handle: image,
            material: material.clone(),
            size: BLOCK_TILE_RES,
            world_size: Vec2::splat(PANEL_SIZE_BLOCKS),
            last_draw_tick: 0,
            redraw_floor_ms: BLOCK_TILE_REDRAW_FLOOR_MS,
            last_draw_at: std::time::Instant::now()
                - std::time::Duration::from_millis(BLOCK_TILE_REDRAW_FLOOR_MS + 1),
        },
        Mesh3d(mesh),
        MeshMaterial3d(material),
        transform,
        GlobalTransform::IDENTITY,
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        Name::new(format!(
            "hud_panel[{},{},{}:f{}]",
            world_block.x, world_block.y, world_block.z, face
        )),
        ChildOf(chunk_entity),
    ));
}

/// Build the tile's local transform so the plane sits flush against
/// the hosting block's face. Rectangle mesh lies in XY with normal +Z;
/// we rotate +Z onto the face outward normal, then translate to the
/// face center + a small offset so the panel reads clearly.
fn panel_face_transform(block_pos_local: IVec3, face: u8) -> Transform {
    let block_center = Vec3::new(
        block_pos_local.x as f32 + 0.5,
        block_pos_local.y as f32 + 0.5,
        block_pos_local.z as f32 + 0.5,
    );
    let normal = face_outward_normal(face);
    let face_center = block_center + normal * (0.5 + PANEL_FACE_OFFSET);
    let rotation = Quat::from_rotation_arc(Vec3::Z, normal);
    Transform {
        translation: face_center,
        rotation,
        scale: Vec3::ONE,
    }
}

fn face_outward_normal(face: u8) -> Vec3 {
    match face {
        0 => Vec3::X,
        1 => Vec3::NEG_X,
        2 => Vec3::Y,
        3 => Vec3::NEG_Y,
        4 => Vec3::Z,
        5 => Vec3::NEG_Z,
        _ => Vec3::Y,
    }
}
