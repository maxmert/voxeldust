//! Celestial body rendering from `WorldState.bodies[]`.
//!
//! The System shard (pre-connected as a secondary on any primary) streams
//! star + planet data as part of its `WorldState.bodies[]`. Each entry has
//! a world-space position (in system units), radius, and colour. This
//! plugin spawns a PBR sphere per body and keeps its transform in sync
//! with the latest snapshot.
//!
//! Source of truth: `WorldState.bodies` is driven by the System secondary;
//! when the System shard updates (e.g. every WorldState tick) we replace
//! the local entity set.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::net_plugin::GameEvent;
use crate::network::NetEvent;
use crate::shard_origin::ShardOrigin;

pub struct CelestialPlugin;

impl Plugin for CelestialPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BodyRegistry>()
            .init_resource::<BodyAssets>()
            .add_systems(Update, ingest_bodies);
    }
}

#[derive(Resource, Default)]
struct BodyRegistry {
    /// `body_id` → Bevy entity.
    entries: HashMap<u32, Entity>,
}

#[derive(Resource, Default)]
struct BodyAssets {
    sphere_mesh: Option<Handle<Mesh>>,
}

fn ingest_bodies(
    mut events: MessageReader<GameEvent>,
    mut registry: ResMut<BodyRegistry>,
    mut assets: ResMut<BodyAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    mut q: Query<&mut Transform>,
    shard_origin: Res<ShardOrigin>,
) {
    // Ensure the shared sphere mesh is built (ico(4) = 642 verts, a good
    // trade-off for distant body rendering).
    if assets.sphere_mesh.is_none() {
        assets.sphere_mesh = Some(meshes.add(Sphere::new(1.0).mesh().ico(4).unwrap()));
    }
    let sphere = assets.sphere_mesh.clone().unwrap();

    for GameEvent(ev) in events.read() {
        // Primary WorldState and any secondary WorldState may carry body
        // data (the System secondary typically does). We accept from either.
        let bodies = match ev {
            NetEvent::WorldState(ws) => &ws.bodies,
            NetEvent::SecondaryWorldState { ws, .. } => &ws.bodies,
            _ => continue,
        };

        for body in bodies {
            let radius = body.radius as f32;
            // Celestial bodies arrive in system-absolute coordinates
            // (billions of metres from the star). The camera/ship is at
            // Bevy origin and represents `shard_origin.origin` in system
            // space. Render at the difference so the body sits at the
            // correct relative direction + distance from the player.
            let rel = body.position - shard_origin.origin;
            let pos = Vec3::new(rel.x as f32, rel.y as f32, rel.z as f32);

            if let Some(&entity) = registry.entries.get(&body.body_id) {
                if let Ok(mut tf) = q.get_mut(entity) {
                    tf.translation = pos;
                    tf.scale = Vec3::splat(radius);
                }
            } else {
                // Spawn a new body entity. Stars get emissive treatment
                // (body_id == 0 is the system star by convention);
                // planets get PBR matte.
                let is_star = body.body_id == 0;
                let material = materials.add(StandardMaterial {
                    base_color: Color::srgb(body.color[0], body.color[1], body.color[2]),
                    emissive: if is_star {
                        LinearRgba::rgb(
                            body.color[0] * 50.0,
                            body.color[1] * 50.0,
                            body.color[2] * 50.0,
                        )
                    } else {
                        LinearRgba::BLACK
                    },
                    perceptual_roughness: if is_star { 0.0 } else { 0.95 },
                    metallic: 0.0,
                    ..default()
                });
                let entity = commands
                    .spawn((
                        Mesh3d(sphere.clone()),
                        MeshMaterial3d(material),
                        Transform::from_translation(pos).with_scale(Vec3::splat(radius)),
                        Name::new(format!("body_{}", body.body_id)),
                    ))
                    .id();
                registry.entries.insert(body.body_id, entity);
                tracing::info!(
                    body_id = body.body_id,
                    is_star,
                    radius,
                    pos = ?pos,
                    "celestial body spawned"
                );
            }
        }
    }
}
