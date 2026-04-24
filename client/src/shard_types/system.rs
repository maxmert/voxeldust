//! SYSTEM shard-type plugin (wire shard_type = 1, scene-context).
//!
//! SYSTEM shards have no blocks of their own. Their rendering
//! contribution is:
//!   * **Celestial body spheres** — stars (emissive PBR) + planets
//!     (matte PBR) at their server-reported positions. Bodies whose
//!     `body_id` has a loaded PLANET shard are skipped (that planet
//!     renders via its own block stream instead) — the reconciliation
//!     sphere ↔ blocks described in the plan's Phase 7.
//!   * **Observable entity → SHIP-secondary linkage** — visible ships
//!     in-system are full SHIP secondaries; their chunks stream in
//!     separately, so we never render placeholders here.
//!
//! Far-field rendering (Design Principle #7): a body more than
//! `FAR_FIELD_THRESHOLD` away is clamped to `FAR_FIELD_RADIUS` along
//! the camera → body direction, with its visible radius scaled by
//! `FAR_FIELD_RADIUS / real_distance` so it preserves apparent angular
//! size. Keeps everything inside f32's safe range without placeholder
//! proxies.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use glam::DVec3;

use voxeldust_core::client_message::WorldStateData;

use crate::shard::{
    plugin::HudSummaryCtx, CameraWorldPos, PrimaryShard, PrimaryWorldState, SecondaryWorldStates,
    ShardKey, ShardOriginSet, ShardRuntime, ShardTypePlugin, ShardTypeRegistry, Secondaries,
};
use crate::shard_types::planet::PLANET_SHARD_TYPE;

pub struct SystemShardPlugin;

impl Plugin for SystemShardPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut()
            .resource_mut::<ShardTypeRegistry>()
            .register(Box::new(SystemShardType));
        app.init_resource::<CelestialBodies>()
            .init_resource::<CelestialAssets>()
            .add_systems(Update, sync_celestial_bodies.after(ShardOriginSet));
    }
}

struct SystemShardType;

impl ShardTypePlugin for SystemShardType {
    fn shard_type(&self) -> u8 {
        SYSTEM_SHARD_TYPE
    }
    fn name(&self) -> &'static str {
        "system"
    }
    fn is_scene_context(&self) -> bool {
        true
    }
    fn to_system_space(&self, _shard: &ShardRuntime, local: DVec3) -> DVec3 {
        local
    }
    fn from_system_space(&self, _shard: &ShardRuntime, world: DVec3) -> DVec3 {
        world
    }
    fn hud_summary(&self, ctx: &HudSummaryCtx) -> Vec<(String, String)> {
        let mut out = Vec::with_capacity(3);
        let ws = match ctx.primary_ws {
            Some(w) => w,
            None => return out,
        };
        out.push(("BODIES".into(), format!("{}", ws.bodies.len())));
        // Nearest body + its distance.
        if let Some((nearest, dist)) = ws
            .bodies
            .iter()
            .map(|b| {
                let pos = DVec3::new(b.position.x, b.position.y, b.position.z);
                (b, (pos - ctx.camera_world).length())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            out.push((
                "NEAREST".into(),
                format!("id={} ({:.0} km)", nearest.body_id, dist / 1000.0),
            ));
        }
        out
    }
}

pub const SYSTEM_SHARD_TYPE: u8 = 1;

// ─────────────────────────────────────────────────────────────────────
// Celestial bodies
// ─────────────────────────────────────────────────────────────────────

/// Below this system-space distance, a body renders at its real
/// position + radius. Beyond it, far-field clamping applies — the
/// body is placed at `FAR_FIELD_RADIUS` along the camera → body
/// direction with its apparent angular size preserved.
const FAR_FIELD_THRESHOLD: f64 = 5.0e5;

/// Radius at which far-field bodies are rendered. f32 precision at
/// this magnitude is sub-mm, safely within render-space bounds.
const FAR_FIELD_RADIUS: f64 = 1.0e6;

/// Spawned sphere entity per live body_id.
#[derive(Resource, Default)]
struct CelestialBodies {
    entities: HashMap<u32, Entity>,
}

/// Cached mesh + material handles so bodies don't re-allocate every
/// time a new body_id appears.
#[derive(Resource, Default)]
struct CelestialAssets {
    star_mesh: Option<Handle<Mesh>>,
    planet_mesh: Option<Handle<Mesh>>,
}

#[derive(Component)]
struct CelestialBody {
    body_id: u32,
}

fn sync_celestial_bodies(
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    primary: Res<PrimaryShard>,
    secondaries: Res<Secondaries>,
    camera_world: Res<CameraWorldPos>,
    mut bodies_resource: ResMut<CelestialBodies>,
    mut assets: ResMut<CelestialAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut existing: Query<(&CelestialBody, &mut Transform, &mut Visibility)>,
    mut commands: Commands,
) {
    // Lazy-init shared meshes.
    if assets.star_mesh.is_none() {
        assets.star_mesh = Some(meshes.add(Sphere::new(1.0).mesh().ico(4).unwrap()));
    }
    if assets.planet_mesh.is_none() {
        assets.planet_mesh = Some(meshes.add(Sphere::new(1.0).mesh().ico(4).unwrap()));
    }

    // Bodies come from the SYSTEM WorldState — primary if we're on
    // SYSTEM, secondary otherwise. If neither present, hide everything
    // and bail (we're not in a scene where bodies should render).
    let source_ws: Option<&WorldStateData> = {
        let on_system_primary = primary
            .current
            .map(|k| k.shard_type == SYSTEM_SHARD_TYPE)
            .unwrap_or(false);
        if on_system_primary {
            primary_ws.latest.as_ref()
        } else {
            secondary_ws.by_shard_type.get(&SYSTEM_SHARD_TYPE)
        }
    };
    let Some(ws) = source_ws else {
        // No SYSTEM data at all — hide any lingering spheres, but
        // keep entities around in case SYSTEM reconnects.
        for (_, _, mut vis) in &mut existing {
            *vis = Visibility::Hidden;
        }
        return;
    };

    // Which body_ids are "owned" by a currently-loaded PLANET shard?
    // They render via their block stream, not as a sphere here.
    // MVP: match the PLANET shard's seed (server uses planet_body_id
    // as the seed convention for PLANET shards; this may need
    // refinement once planet-as-blocks lands). For now we treat every
    // loaded PLANET shard as excluding a body at its seed.
    let owned_by_planet: HashSet<u32> = secondaries
        .runtimes
        .keys()
        .filter(|k| k.shard_type == PLANET_SHARD_TYPE)
        .map(|k| k.seed as u32)
        .collect();

    // Set-diff against last tick's spawned body ids.
    let mut seen: HashSet<u32> = HashSet::with_capacity(ws.bodies.len());
    for body in &ws.bodies {
        seen.insert(body.body_id);
        if owned_by_planet.contains(&body.body_id) {
            // Skip — a PLANET shard is rendering this body's blocks.
            if let Some(&e) = bodies_resource.entities.get(&body.body_id) {
                if let Ok((_, _, mut vis)) = existing.get_mut(e) {
                    *vis = Visibility::Hidden;
                }
            }
            continue;
        }
        render_body(
            body,
            camera_world.pos,
            &assets,
            &mut materials,
            &mut bodies_resource,
            &mut existing,
            &mut commands,
        );
    }

    // Despawn bodies that disappeared from WS.bodies[] (e.g., server
    // destroyed a body, or we left a system on warp).
    let stale: Vec<u32> = bodies_resource
        .entities
        .keys()
        .filter(|id| !seen.contains(id))
        .copied()
        .collect();
    for id in stale {
        if let Some(e) = bodies_resource.entities.remove(&id) {
            commands.entity(e).despawn();
        }
    }
}

fn render_body(
    body: &voxeldust_core::client_message::CelestialBodyData,
    cam_world: DVec3,
    assets: &CelestialAssets,
    materials: &mut Assets<StandardMaterial>,
    bodies_resource: &mut CelestialBodies,
    existing: &mut Query<(&CelestialBody, &mut Transform, &mut Visibility)>,
    commands: &mut Commands,
) {
    // Far-field vs near-field. Compute once per body per tick.
    let body_pos = DVec3::new(body.position.x, body.position.y, body.position.z);
    let rel = body_pos - cam_world;
    let dist = rel.length();
    let (render_pos_f32, render_radius_f32) = if dist > FAR_FIELD_THRESHOLD {
        // Far-field: clamp distance, scale apparent radius to preserve
        // angular size. `dir` is safe because dist > threshold > 0.
        let dir = rel / dist;
        let clamped = dir * FAR_FIELD_RADIUS;
        let scaled_radius = body.radius * FAR_FIELD_RADIUS / dist;
        (
            Vec3::new(clamped.x as f32, clamped.y as f32, clamped.z as f32),
            scaled_radius as f32,
        )
    } else {
        (
            Vec3::new(rel.x as f32, rel.y as f32, rel.z as f32),
            body.radius as f32,
        )
    };

    let is_star = body.body_id == 0;
    if let Some(&entity) = bodies_resource.entities.get(&body.body_id) {
        if let Ok((_, mut tf, mut vis)) = existing.get_mut(entity) {
            tf.translation = render_pos_f32;
            tf.scale = Vec3::splat(render_radius_f32);
            *vis = Visibility::Inherited;
        }
    } else {
        let material = materials.add(body_material(is_star, body.color));
        let mesh = if is_star {
            assets.star_mesh.clone().unwrap()
        } else {
            assets.planet_mesh.clone().unwrap()
        };
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::from_translation(render_pos_f32)
                    .with_scale(Vec3::splat(render_radius_f32)),
                Visibility::default(),
                Name::new(format!(
                    "celestial_body[{}]",
                    if is_star { "star".to_string() } else { format!("{}", body.body_id) }
                )),
                CelestialBody { body_id: body.body_id },
            ))
            .id();
        bodies_resource.entities.insert(body.body_id, entity);
    }
}

fn body_material(is_star: bool, color: [f32; 3]) -> StandardMaterial {
    if is_star {
        StandardMaterial {
            base_color: Color::srgb(color[0], color[1], color[2]),
            emissive: LinearRgba::rgb(color[0] * 40.0, color[1] * 40.0, color[2] * 40.0),
            unlit: true,
            ..default()
        }
    } else {
        StandardMaterial {
            base_color: Color::srgb(color[0], color[1], color[2]),
            perceptual_roughness: 0.85,
            metallic: 0.0,
            ..default()
        }
    }
}

// ShardKey is needed by the helper functions above through import —
// keeping the use statement explicit here avoids breaking if the parent
// module re-shuffles.
#[allow(dead_code)]
fn _shard_key_ref(_k: ShardKey) {}
