//! Wireframe-cube block highlight. Lazily spawns one reusable entity
//! parented under the hit's `ChunkSource`. Hidden when no hit.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};

use crate::interaction::raycast::{BlockRaycastSet, BlockTarget};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockHighlightSet;

pub struct BlockHighlightPlugin;

impl Plugin for BlockHighlightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HighlightHandles>().add_systems(
            Update,
            update_highlight.in_set(BlockHighlightSet).after(BlockRaycastSet),
        );
    }
}

#[derive(Component)]
struct BlockHighlight;

#[derive(Resource, Default)]
struct HighlightHandles {
    mesh: Option<Handle<Mesh>>,
    material: Option<Handle<StandardMaterial>>,
    entity: Option<Entity>,
}

fn update_highlight(
    target: Res<BlockTarget>,
    mut handles: ResMut<HighlightHandles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    mut q: Query<(&mut Transform, &mut Visibility), With<BlockHighlight>>,
) {
    // Lazy init: wireframe-cube mesh + unlit emissive material.
    if handles.mesh.is_none() {
        handles.mesh = Some(meshes.add(wireframe_cube()));
    }
    if handles.material.is_none() {
        handles.material = Some(materials.add(StandardMaterial {
            base_color: Color::srgba(0.8, 0.9, 1.0, 1.0),
            emissive: LinearRgba::rgb(0.8, 0.9, 1.0),
            unlit: true,
            cull_mode: None,
            ..default()
        }));
    }

    let Some(hit) = target.hit else {
        // Hide existing highlight if any.
        if let Some(e) = handles.entity {
            if let Ok((_, mut vis)) = q.get_mut(e) {
                *vis = Visibility::Hidden;
            }
        }
        return;
    };

    // Ensure we have an entity to drive.
    let entity = match handles.entity {
        Some(e) if commands.get_entity(e).is_ok() => e,
        _ => {
            let e = commands
                .spawn((
                    Mesh3d(handles.mesh.clone().unwrap()),
                    MeshMaterial3d(handles.material.clone().unwrap()),
                    Transform::IDENTITY,
                    Visibility::default(),
                    Name::new("block_highlight"),
                    BlockHighlight,
                ))
                .id();
            handles.entity = Some(e);
            e
        }
    };

    // Re-parent + move the highlight to the hit. `ChildOf` is an
    // immutable component in Bevy 0.18 — mutate via `commands.insert`
    // which re-inserts the component wholesale.
    commands.entity(entity).insert(ChildOf(hit.source_entity));
    if let Ok((mut tf, mut vis)) = q.get_mut(entity) {
        tf.translation = Vec3::new(
            hit.block_pos.x as f32 + 0.5,
            hit.block_pos.y as f32 + 0.5,
            hit.block_pos.z as f32 + 0.5,
        );
        tf.scale = Vec3::splat(1.002);
        *vis = Visibility::Inherited;
    }
}

fn wireframe_cube() -> Mesh {
    // 12 unit-cube edges as a line list. Centered on origin in a 1m³
    // cube, so `Transform.translation = block_pos + 0.5` places it
    // snugly around the target block.
    let v = |x: f32, y: f32, z: f32| [x - 0.5, y - 0.5, z - 0.5];
    let positions: Vec<[f32; 3]> = vec![
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(1.0, 0.0, 1.0),
        v(0.0, 0.0, 1.0),
        v(0.0, 1.0, 0.0),
        v(1.0, 1.0, 0.0),
        v(1.0, 1.0, 1.0),
        v(0.0, 1.0, 1.0),
    ];
    let indices: Vec<u32> = vec![
        // bottom
        0, 1, 1, 2, 2, 3, 3, 0,
        // top
        4, 5, 5, 6, 6, 7, 7, 4,
        // pillars
        0, 4, 1, 5, 2, 6, 3, 7,
    ];
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
