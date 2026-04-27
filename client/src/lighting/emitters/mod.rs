//! Block-driven local lighting — Phase 8.
//!
//! For every placed block (full-block or sub-block) whose registry
//! entry carries a [`voxeldust_core::block::registry::LightSpec`], we
//! spawn a child `PointLight` or `SpotLight` parented to the chunk
//! entity, plus an HDR-emissive proxy mesh so the fixture face itself
//! reads as a bright source under Bloom + AgX. Photometric properties
//! (lumens, blackbody temperature, beam shape, range, shadow class)
//! come from the shared block registry — identical on server and
//! client by construction, so a placed lamp behaves the same on every
//! connected client.
//!
//! ## Module layout
//!
//! - [`full_block`] — full-block emitters (thrusters, reactors). One
//!   emitter occupies an entire 1 m voxel; iteration walks the chunk's
//!   62³ cells.
//! - [`sub_block`] — sub-block emitters (surface lamps via
//!   `SubBlockType::SurfaceLight` and friends). One emitter sits flush
//!   against a host-block face.
//! - [`throttle`] — cross-cutting infrastructure for signal-modulated
//!   intensity (thrusters dim/light with their throttle channel; future
//!   functional lamps will subscribe through the same mechanism).
//!
//! `mod.rs` (this file) owns the per-spawn-pass shared infrastructure:
//! geometry helpers, the `LocalShadowBudget`, the public chunk-spawn
//! entry, and the `LocalLightingPlugin` that registers the throttle
//! systems. Per-block-kind modules call back into the helpers exposed
//! here.
//!
//! ## Lifecycle
//!
//! `spawn_local_lights_for_chunk` is invoked from
//! `client/src/chunk/stream.rs::remesh_chunk_from_cache` after the
//! main mesh + sub-block + HUD-tile children are spawned. Because the
//! chunk entity is recursively despawned on each remesh (snapshot or
//! delta), the existing `ChildOf(chunk_entity)` parenting cascades
//! light despawns automatically. Re-spawning from scratch on every
//! remesh is the same pattern HUD tiles use; the cost (one
//! `commands.spawn` per light) is negligible compared to the mesh
//! rebuild itself.
//!
//! ## Photometry
//!
//! Bevy 0.18's `PointLight.intensity` and `SpotLight.intensity` are in
//! lumens (luminous flux Φ). The `LightSpec.lumens` field flows through
//! unchanged. Colour is `blackbody(K) × tint`, computed via
//! [`voxeldust_core::blackbody::temperature_to_linear_rgb`] — no magic-
//! number RGB tables.
//!
//! ## Shadow budget
//!
//! `LightingFidelity.max_shadow_casting_local_lights` caps how many
//! point/spot lights may cast shadows simultaneously. Lights whose
//! `shadow_caster_class` is `0` always run shadowless (cheap fill);
//! `>= 1` lights cast shadows up to the budget, allocated greedily in
//! iteration order (chunk → block-position scan). The cap is per-spawn-
//! pass so a ship with hundreds of lamps stays inside the directional-
//! light + spot-light shadow-map budget the GPU can comfortably
//! sustain.

pub mod full_block;
pub mod lamp_configs;
pub mod sub_block;
pub mod throttle;

use bevy::color::LinearRgba;
use bevy::light::{PointLight, SpotLight};
use bevy::math::primitives::{Cuboid, Rectangle};
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;

use voxeldust_core::block::block_meta::BlockOrientation;
use voxeldust_core::block::chunk_storage::ChunkStorage;
use voxeldust_core::block::palette::CHUNK_SIZE;
use voxeldust_core::block::registry::{BeamKind, BlockRegistry, EmissiveGeometry, LightSpec};
use voxeldust_core::blackbody::temperature_to_linear_rgb;

use crate::config::GameConfig;

pub use lamp_configs::LampConfigs;
pub use throttle::{
    ThrottleModulatedLight, ThrottleModulatedProxy, ThrottleSignals,
};

// ──────────────────────────────────────────────────────────────────────────
// Plugin
// ──────────────────────────────────────────────────────────────────────────

pub struct LocalLightingPlugin;

impl Plugin for LocalLightingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(lamp_configs::LampConfigsPlugin)
            .init_resource::<ThrottleSignals>()
            .add_systems(
                Update,
                (
                    throttle::update_throttle_signals,
                    throttle::apply_throttle_modulation,
                )
                    .chain(),
            );
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Public chunk-spawn entry
// ──────────────────────────────────────────────────────────────────────────

/// Local-light spawn pass for one chunk. Called from the chunk-remesh
/// path *after* the chunk entity has been spawned and before any
/// subsequent system queries the chunk's children.
///
/// Internally calls into [`full_block`] and [`sub_block`] in turn;
/// each module returns `(spawned, shadow_casters)` so the diagnostic
/// log line below carries the correct totals.
///
/// `shard` + `chunk_index` are forwarded to `sub_block` so it can look
/// up player-customised lamp configs in [`LampConfigs`] keyed by
/// `(shard, world_pos, face)`. Pass `None` for `shard` when the chunk
/// can't be associated with one (the broadcast pre-shard-resolution
/// path); custom configs are silently skipped in that case.
#[allow(clippy::too_many_arguments)]
pub fn spawn_local_lights_for_chunk(
    commands: &mut Commands,
    config: &GameConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    chunk_entity: Entity,
    chunk: &ChunkStorage,
    registry: &BlockRegistry,
    shadow_budget: &mut LocalShadowBudget,
    lamp_configs: &LampConfigs,
    shard: Option<crate::shard::ShardKey>,
    chunk_index: bevy::math::IVec3,
) {
    let _ = config; // future: per-shard / per-quality scaling
    // Cache the proxy meshes so all emissive proxies share two
    // handles — Bevy refcounts mesh assets, so this is essentially free
    // after the first emitter.
    let cube_mesh: Handle<Mesh> = meshes.add(Cuboid::new(
        EMISSIVE_PROXY_SIDE,
        EMISSIVE_PROXY_SIDE,
        EMISSIVE_PROXY_SIDE,
    ));
    let face_mesh: Handle<Mesh> =
        meshes.add(Rectangle::new(EMISSIVE_FACE_SIDE, EMISSIVE_FACE_SIDE));

    let (full_spawned, full_shadows) = full_block::spawn_full_block_emitters(
        commands,
        materials,
        chunk_entity,
        chunk,
        registry,
        shadow_budget,
        &cube_mesh,
        &face_mesh,
    );
    let (sub_spawned, sub_shadows) = sub_block::spawn_sub_block_emitters(
        commands,
        materials,
        chunk_entity,
        chunk,
        registry,
        shadow_budget,
        &cube_mesh,
        &face_mesh,
        lamp_configs,
        shard,
        chunk_index,
    );
    let spawned = full_spawned + sub_spawned;
    let shadow_casters = full_shadows + sub_shadows;
    if spawned > 0 {
        tracing::info!(
            spawned,
            shadow_casters,
            "lighting/local: spawned per-block lights for chunk"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Shadow budget
// ──────────────────────────────────────────────────────────────────────────

/// Per-spawn-pass shadow-casting-light budget. Construct once at the
/// start of a remesh pass with the active `LightingFidelity` budget;
/// every `try_consume()` succeeds while there's slack, then returns
/// `false`.
pub struct LocalShadowBudget {
    /// Remaining slots before lights stop casting shadows.
    remaining: u32,
}

impl LocalShadowBudget {
    pub fn new(config: &GameConfig) -> Self {
        Self {
            remaining: config.graphics.lighting.max_shadow_casting_local_lights,
        }
    }

    /// Try to consume one shadow-casting slot. Returns `true` if a slot
    /// was consumed (caller should enable shadows on the light); `false`
    /// if the budget is full (caller should spawn the light shadowless).
    pub(super) fn try_consume(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Geometry constants
// ──────────────────────────────────────────────────────────────────────────

/// Block-side-length (m). Each block is a unit cube; the centre is at
/// `+0.5` on each axis from the lower corner.
pub(super) const BLOCK_CENTRE_OFFSET: f32 = 0.5;

/// `CHUNK_SIZE` re-exported as `u8` for the inner iteration. Compile-
/// time constant; the cast is checked.
pub(super) const CHUNK_SIZE_U8: u8 = CHUNK_SIZE as u8;

/// Side length (m) of the emissive cube spawned inside each light-
/// emitter block. The proxy must protrude **past** the chunk mesh's
/// matte block faces so the bright pixels actually appear on screen.
/// 1.02 m gives a 1 cm overshoot per face — enough to win the depth
/// test on every air-facing side, while the side faces that border
/// adjacent hull cubes remain occluded by the neighbour's mesh face.
const EMISSIVE_PROXY_SIDE: f32 = 1.02;

/// Edge length (m) of the directional-face proxy quad. 0.95 m leaves
/// a 2.5 cm border between the glow and the neighbouring block's mesh
/// face so the bulb-rim reads as a recessed nozzle / lens.
const EMISSIVE_FACE_SIDE: f32 = 0.95;

/// How far (m) outside the block boundary the directional-face proxy
/// sits along its facing normal. 0.51 m places the quad's plane just
/// outside the 0.5 m-from-centre block face — winning the depth test
/// over the chunk mesher's matte face without colliding with any
/// neighbour block.
const EMISSIVE_FACE_OUTSET: f32 = 0.51;

/// Reference radiance (W/m²) at which the emissive output equals the
/// per-light spec colour without amplification.
const EMISSIVE_REFERENCE_RADIANCE_W_PER_M2: f32 = 2_000.0;

/// Extra HDR multiplier applied to the emissive output so it reaches
/// the bloom threshold cleanly. 4.0 lifts a 12 000 W/m² LAMP_WHITE
/// proxy to ≈ 24 × HDR — well past AgX's clip-to-white knee.
const HDR_BLOOM_MULTIPLIER: f32 = 4.0;

// ──────────────────────────────────────────────────────────────────────────
// Shared spawn helpers
// ──────────────────────────────────────────────────────────────────────────

/// Compute the per-light linear-sRGB colour from `(blackbody_K, tint)`.
pub(super) fn blackbody_with_tint(temperature_k: f32, tint: [f32; 3]) -> Color {
    let bb = temperature_to_linear_rgb(temperature_k as f64);
    Color::linear_rgb(bb[0] * tint[0], bb[1] * tint[1], bb[2] * tint[2])
}

/// Spawn one `PointLight` or `SpotLight` child of the chunk entity.
/// If `throttle_channel` is `Some`, attaches a `ThrottleModulatedLight`
/// component so the per-frame `apply_throttle_modulation` system scales
/// `intensity` by the broadcast signal value.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_one_light(
    commands: &mut Commands,
    chunk_entity: Entity,
    local: Vec3,
    spec: &LightSpec,
    color: Color,
    shadows_enabled: bool,
    throttle_channel: Option<&'static str>,
) {
    let mut entity_commands = match spec.beam_kind {
        BeamKind::PointOmni => commands.spawn((
            PointLight {
                color,
                intensity: spec.lumens,
                range: spec.range_m,
                shadows_enabled,
                ..PointLight::default()
            },
            Transform::from_translation(local),
            Name::new("local_point_light"),
            ChildOf(chunk_entity),
        )),
        BeamKind::SpotCone {
            inner_angle_rad,
            outer_angle_rad,
        } => commands.spawn((
            SpotLight {
                color,
                intensity: spec.lumens,
                range: spec.range_m,
                inner_angle: inner_angle_rad,
                outer_angle: outer_angle_rad,
                shadows_enabled,
                ..SpotLight::default()
            },
            Transform::from_translation(local),
            Name::new("local_spot_light"),
            ChildOf(chunk_entity),
        )),
    };
    if let Some(channel) = throttle_channel {
        entity_commands.insert(ThrottleModulatedLight {
            channel: channel.to_string(),
            base_intensity: spec.lumens,
        });
    }
}

/// Spawn the visible emissive surface for a light-emitter block.
///
/// Two geometry modes (chosen via `LightSpec.emissive_geometry`):
///   * `FullCube` — slightly outset cube glowing on every visible side.
///   * `DirectionalFace` — single flat quad just outside the block's
///     facing direction.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_emissive_proxy(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    chunk_entity: Entity,
    local: Vec3,
    spec: &LightSpec,
    color: Color,
    orientation: BlockOrientation,
    cube_mesh: Handle<Mesh>,
    face_mesh: Handle<Mesh>,
    throttle_channel: Option<&'static str>,
) -> Handle<StandardMaterial> {
    let linear = color.to_linear();
    let radiance_factor =
        spec.emissive_radiance_w_per_m2 / EMISSIVE_REFERENCE_RADIANCE_W_PER_M2;
    let hdr = radiance_factor * HDR_BLOOM_MULTIPLIER;
    let emissive = LinearRgba::new(
        linear.red * hdr,
        linear.green * hdr,
        linear.blue * hdr,
        1.0,
    );
    let mat_handle = materials.add(StandardMaterial {
        unlit: true,
        base_color: color,
        emissive,
        ..StandardMaterial::default()
    });

    let mut entity_commands = match spec.emissive_geometry {
        EmissiveGeometry::FullCube => commands.spawn((
            Mesh3d(cube_mesh),
            MeshMaterial3d(mat_handle.clone()),
            Transform::from_translation(local),
            Name::new("local_light_emissive_cube"),
            ChildOf(chunk_entity),
        )),
        EmissiveGeometry::DirectionalFace => {
            let facing = facing_unit_vec3(orientation);
            let translation = local + facing * EMISSIVE_FACE_OUTSET;
            let rotation = Quat::from_rotation_arc(Vec3::Z, facing);
            commands.spawn((
                Mesh3d(face_mesh),
                MeshMaterial3d(mat_handle.clone()),
                Transform {
                    translation,
                    rotation,
                    ..Transform::default()
                },
                Name::new("local_light_emissive_face"),
                ChildOf(chunk_entity),
            ))
        }
    };
    if let Some(channel) = throttle_channel {
        entity_commands.insert(ThrottleModulatedProxy {
            channel: channel.to_string(),
            base_emissive: emissive,
        });
    }
    mat_handle
}

/// Convert a `BlockOrientation`'s `facing()` index (0..6) into a unit
/// `Vec3` — `core::block::block_meta` exposes the same as `DVec3`, but
/// Bevy's `Transform` uses single-precision so we collapse here.
pub(super) fn facing_unit_vec3(orientation: BlockOrientation) -> Vec3 {
    match orientation.facing() {
        0 => Vec3::X,
        1 => Vec3::NEG_X,
        2 => Vec3::Y,
        3 => Vec3::NEG_Y,
        4 => Vec3::Z,
        _ => Vec3::NEG_Z,
    }
}
