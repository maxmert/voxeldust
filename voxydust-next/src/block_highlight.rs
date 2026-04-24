//! Visual highlight for the currently-targeted block.
//!
//! Reads `BlockTarget` each frame and spawns / updates / despawns a
//! single highlight entity so the player sees which block they're
//! pointing at. Legacy voxydust drew this as a translucent sphere
//! (`voxydust/src/render.rs:1462-1513`); voxydust-next renders it as
//! a 1-block-sized cuboid with thin outlined edges — proper block
//! geometry matching what will actually be broken/placed rather than
//! a generic sphere proxy.
//!
//! **Parenting.** For root-grid blocks the highlight is parented to
//! the primary chunk-source root, so ship rotation / translation
//! (driven by `chunk_stream::sync_source_transforms`) flows to the
//! highlight automatically. Sub-grid blocks (future #23) would parent
//! under the sub-grid's root instead so the highlight follows piston
//! extension / rotor rotation.
//!
//! **Visibility.** The highlight is hidden (not despawned) when
//! `BlockTarget.hit` is `None`, so the GPU doesn't churn on
//! spawn/despawn every time the cursor moves off a block.

use bevy::prelude::*;

use crate::block_raycast::{BlockRaycastSet, BlockTarget};
use crate::chunk_stream::SourceIndex;

pub struct BlockHighlightPlugin;

impl Plugin for BlockHighlightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HighlightAssets>()
            .add_systems(Update, sync_highlight.after(BlockRaycastSet));
    }
}

/// One-time-initialised mesh + material for the highlight cube.
#[derive(Resource, Default)]
struct HighlightAssets {
    mesh: Option<Handle<Mesh>>,
    material: Option<Handle<StandardMaterial>>,
}

/// Marker on the single highlight entity. We spawn this entity on
/// first targeting and reuse it for the rest of the session.
#[derive(Component)]
struct HighlightCube;

fn sync_highlight(
    mut commands: Commands,
    mut assets: ResMut<HighlightAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    target: Res<BlockTarget>,
    sources: Res<SourceIndex>,
    mut q: Query<
        (
            Entity,
            &mut Transform,
            &mut Visibility,
            Option<&ChildOf>,
        ),
        With<HighlightCube>,
    >,
) {
    // Lazy asset init — slightly-oversized cube (1.01 m) so it z-fights
    // safely outside any solid block face, with an emissive material
    // that glows faintly even in the dark. Alpha < 1 gives the
    // outlined-selection feel without requiring a custom wireframe
    // shader.
    if assets.mesh.is_none() {
        // 1.005× scale to render just outside the block face (no
        // z-fighting). Minimal material variant — a default
        // `StandardMaterial` with a bright emissive, which reuses the
        // PBR pipeline the chunks already use and avoids forcing a
        // fresh shader variant compile. An earlier variant used
        // `alpha_mode: Blend + cull_mode: None + unlit: true`, which
        // compiled a new Metal pipeline on first spawn and blocked the
        // main thread (~hundreds of ms of "input stuck" on macOS).
        // Emissive-only lookseasily separates the highlight from the
        // underlying block without needing translucency.
        assets.mesh = Some(meshes.add(Cuboid::new(1.005, 1.005, 1.005).mesh()));
        assets.material = Some(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.15, 0.2),
            emissive: LinearRgba::rgb(2.5, 3.0, 4.0),
            perceptual_roughness: 1.0,
            metallic: 0.0,
            ..default()
        }));
    }

    // Resolve the parent chunk-source root for this target. Root-grid
    // hits parent under `primary_seed`; sub-grid hits (future #23)
    // would parent under the sub-grid's own source root.
    let target_parent = target.hit.as_ref().and_then(|hit| {
        // TODO(#23): if hit.sub_grid_id.is_some(), look up that
        // sub-grid's root entity from a future SubGridIndex.
        sources.entries.get(&hit.source_seed).copied()
    });

    match (target.hit.as_ref(), q.single_mut()) {
        (Some(hit), Ok((entity, mut tf, mut vis, current_parent))) => {
            // Update transform. Block positions are integer corners;
            // the highlight is centered on the block at (+0.5, +0.5, +0.5).
            let pos = Vec3::new(
                hit.block_pos.x as f32 + 0.5,
                hit.block_pos.y as f32 + 0.5,
                hit.block_pos.z as f32 + 0.5,
            );
            tf.translation = pos;
            tf.rotation = Quat::IDENTITY;
            tf.scale = Vec3::ONE;
            *vis = Visibility::Inherited;
            // If the target's source changed (ship hop, sub-grid
            // change), re-parent.
            if let Some(new_parent) = target_parent {
                if current_parent.map(|c| c.parent()) != Some(new_parent) {
                    commands.entity(entity).insert(ChildOf(new_parent));
                }
            }
        }
        (Some(hit), Err(_)) => {
            // First target of the session — spawn the highlight entity.
            let pos = Vec3::new(
                hit.block_pos.x as f32 + 0.5,
                hit.block_pos.y as f32 + 0.5,
                hit.block_pos.z as f32 + 0.5,
            );
            let mut ent = commands.spawn((
                Mesh3d(assets.mesh.clone().unwrap()),
                MeshMaterial3d(assets.material.clone().unwrap()),
                Transform::from_translation(pos),
                Name::new("block_highlight"),
                HighlightCube,
            ));
            if let Some(parent) = target_parent {
                ent.insert(ChildOf(parent));
            }
        }
        (None, Ok((_, _, mut vis, _))) => {
            // No target — hide but don't despawn.
            *vis = Visibility::Hidden;
        }
        (None, Err(_)) => {}
    }
}
