//! Starfield rendering — one mesh, one entity, one draw call.
//!
//! Replaces the per-entity sphere-spam from the previous
//! `galaxy_render` module. All visible stars from `StarCatalog` are
//! baked into a single triangle mesh of camera-facing quads, rendered
//! through the custom `StarfieldMaterial`. The shader
//! (`assets/shaders/starfield.wgsl`) lerps between two positioning
//! schemes via a uniform:
//!
//!   * **Skybox** — fixed distance × anchor-relative direction. Camera
//!     position has no effect on apparent star bearing. This is the
//!     "pane" used in SHIP / SYSTEM / PLANET scenes.
//!   * **Parallax** — real position relative to the camera (clamped to
//!     1.5× skybox radius). Used while GALAXY shard is primary so the
//!     player sees stars sweep past during warp travel.
//!
//! Mode is driven by `PrimaryShard.shard_type`: GALAXY (3) → 1.0,
//! everything else → 0.0, lerped over ~0.8s for a smooth transition.
//!
//! The mesh's vertex positions are stored relative to a `mesh_anchor`
//! `DVec3` in galaxy space so f32 stays precise. As the camera moves
//! through galaxy space (only meaningful in parallax mode), the
//! per-frame `cam_offset = camera_galaxy_pos − mesh_anchor` grows;
//! when it exceeds half the skybox radius we re-anchor and rebuild
//! the mesh (cheap — ~1 ms for 50k stars).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{
    Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat,
};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use glam::DVec3;

use voxeldust_core::client_message::{StarCatalogData, StarCatalogEntryData};
use voxeldust_core::galaxy::GALAXY_UNIT_IN_BLOCKS;

use crate::net::{GameEvent, NetEvent};
use crate::shard::CameraWorldPos;

// ──────────────────────────────────────────────────────────────────────────
// Tunables — same semantics as the previous galaxy_render constants.
// ──────────────────────────────────────────────────────────────────────────

/// Skybox / parallax-clamp radius, in metres. Stars in skybox mode
/// sit exactly here; parallax clamps to 1.5× this. Stays well inside
/// the camera's far plane (`2.0e6` in `main.rs:182`).
const SKYBOX_RADIUS_M: f32 = 1.0e6;

/// Stars dimmer than this (L☉-relative) are dropped — same threshold
/// the per-entity path used. ~0.05 L☉ ≈ a bright M0 dwarf and still
/// removes ~half the catalogue from a typical galaxy.
const MIN_VISIBLE_LUMINOSITY: f32 = 0.05;

/// Apparent half-size of a 1 L☉ star at `SKYBOX_RADIUS_M`, in metres.
/// Cube-rooted with luminosity per-star so brightness range stays
/// readable. 1500 m / 1e6 m ≈ 0.086° angular diameter — reads as a
/// 2-3 px point at 1080p / 60° FOV.
const STAR_BASE_HALF_SIZE_M: f32 = 1500.0;

/// Re-anchor the mesh when |cam_offset| exceeds this. Half the skybox
/// radius keeps each star's anchor-relative position well inside f32
/// precision and bounds the lerp distance during transitions.
const REANCHOR_THRESHOLD_M: f32 = 0.5e6;

/// Seconds for the skybox↔parallax crossfade.
const TRANSITION_SECONDS: f32 = 0.8;

/// `warp_phase` byte values emitted by the galaxy shard
/// (`galaxy-shard/src/main.rs:923-928`). Don't share encoding with
/// `voxeldust_core::autopilot::FlightPhase` — the galaxy shard maps
/// only the warp phases to a custom byte set.
const WARP_PHASE_CRUISE: u8 = 22;
const WARP_PHASE_DECELERATE: u8 = 23;

// ──────────────────────────────────────────────────────────────────────────
// Resources
// ──────────────────────────────────────────────────────────────────────────

/// Authoritative star catalogue, received once per galaxy-shard
/// connect (or generated locally from the seed as a fallback).
/// Replaced wholesale on a new galaxy_seed.
#[derive(Resource, Default, Debug, Clone)]
pub struct StarCatalog {
    pub galaxy_seed: u64,
    pub stars: Vec<StarCatalogEntryData>,
}

/// Player / camera position in galaxy units (f64). Updated from the
/// galaxy shard's `GalaxyWorldState` while warping; anchored on the
/// current system's star while in-system so the sky doesn't snap on
/// warp entry.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct GalaxyCameraPos {
    pub pos_galaxy_units: DVec3,
}

/// Latest warp phase byte from `GalaxyWorldState`. Drives the
/// parallax mode crossfade — see `update_starfield_mode`. The galaxy
/// shard streams these every tick while warping (even when the
/// galaxy shard is a *secondary* under SHIP-primary), so this is a
/// reliable warp signal regardless of which shard is primary.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct WarpPhase {
    pub phase: u8,
}

/// Smoothed parallax amount fed to the shader uniform.
/// 0.0 = skybox, 1.0 = full parallax. Lerps toward `target` at
/// `1.0 / TRANSITION_SECONDS` per second.
#[derive(Resource, Debug, Clone, Copy)]
pub struct StarfieldMode {
    pub parallax_amount: f32,
    pub target: f32,
}

impl Default for StarfieldMode {
    fn default() -> Self {
        Self {
            parallax_amount: 0.0,
            target: 0.0,
        }
    }
}

/// Tracks the live mesh + material handles, plus the galaxy-space
/// anchor used when the mesh was last built. The starfield is a
/// single Bevy entity (with `StarfieldEntity` marker); we rebuild its
/// mesh in-place on catalog change or anchor advance.
#[derive(Resource, Default, Debug)]
pub struct StarfieldHandles {
    pub mesh: Option<Handle<Mesh>>,
    pub material: Option<Handle<StarfieldMaterial>>,
    pub anchor_galaxy_units: DVec3,
    /// `galaxy_seed` the live mesh was built for. Mismatch → rebuild.
    pub built_for_seed: u64,
    /// Number of visible stars in the live mesh — for diagnostics.
    pub star_count: u32,
}

/// Marker on the singleton starfield render entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct StarfieldEntity;

// ──────────────────────────────────────────────────────────────────────────
// Custom mesh attribute for the per-vertex (corner, size, _) packet.
// ──────────────────────────────────────────────────────────────────────────

/// `xy` = quad corner ∈ {-1,+1}², `z` = apparent half-size in metres,
/// `w` reserved. ID `1041` is unused by the standard attributes
/// (those occupy 0–7).
const STARFIELD_PARAMS_ATTR: MeshVertexAttribute =
    MeshVertexAttribute::new("Starfield_Params", 1041, VertexFormat::Float32x4);

// ──────────────────────────────────────────────────────────────────────────
// Custom material
// ──────────────────────────────────────────────────────────────────────────

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct StarfieldUniform {
    /// `camera_galaxy_pos − mesh_anchor`, in metres.
    pub cam_offset: Vec3,
    pub parallax_amount: f32,
    pub skybox_radius: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct StarfieldMaterial {
    #[uniform(0)]
    pub uniform: StarfieldUniform,
}

impl Material for StarfieldMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/starfield.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/starfield.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        // Pure additive; the shader emits alpha = 0 so
        // BLEND_PREMULTIPLIED_ALPHA collapses to `dst.rgb += rgb`.
        AlphaMode::Add
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(1),
            STARFIELD_PARAMS_ATTR.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Plugin
// ──────────────────────────────────────────────────────────────────────────

pub struct StarfieldPlugin;

impl Plugin for StarfieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarfieldMaterial>::default())
            .init_resource::<StarCatalog>()
            .init_resource::<GalaxyCameraPos>()
            .init_resource::<WarpPhase>()
            .init_resource::<StarfieldMode>()
            .init_resource::<StarfieldHandles>()
            .add_systems(
                Update,
                (
                    ingest_star_catalog,
                    ingest_galaxy_world_state,
                    update_starfield_mode,
                    rebuild_starfield_mesh,
                    update_starfield_uniform,
                )
                    .chain(),
            );
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Ingest systems (catalog + camera) — behaviour preserved from
// galaxy_render.rs.
// ──────────────────────────────────────────────────────────────────────────

/// Drain `NetEvent::StarCatalog` into the `StarCatalog` resource.
/// On `Connected` with a non-zero galaxy_seed and an empty catalog,
/// generate one locally from the seed (deterministic; matches the
/// server's `GalaxyMap::generate`). Server-sent catalogues still
/// override.
fn ingest_star_catalog(
    mut events: MessageReader<GameEvent>,
    mut catalog: ResMut<StarCatalog>,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::StarCatalog(data) => {
                let StarCatalogData { galaxy_seed, stars } = data.clone();
                catalog.galaxy_seed = galaxy_seed;
                catalog.stars = stars;
                tracing::info!(
                    galaxy_seed = catalog.galaxy_seed,
                    stars = catalog.stars.len(),
                    source = "server",
                    "starfield: catalog ingested",
                );
            }
            NetEvent::Connected { galaxy_seed, .. } => {
                if *galaxy_seed != 0 && catalog.stars.is_empty() {
                    let map = voxeldust_core::galaxy::GalaxyMap::generate(*galaxy_seed);
                    catalog.galaxy_seed = *galaxy_seed;
                    catalog.stars = map
                        .stars
                        .iter()
                        .enumerate()
                        .map(|(i, s)| StarCatalogEntryData {
                            index: i as u32,
                            position: s.position,
                            system_seed: s.system_seed,
                            star_class: s.star_class as u8,
                            luminosity: s.luminosity as f32,
                        })
                        .collect();
                    tracing::info!(
                        galaxy_seed = catalog.galaxy_seed,
                        stars = catalog.stars.len(),
                        source = "local-from-seed",
                        "starfield: catalog ingested",
                    );
                }
            }
            _ => {}
        }
    }
}

/// Update `GalaxyCameraPos` from authoritative warp updates, with a
/// fallback that anchors on the current system's star at connect time
/// so the sky doesn't snap when warp begins.
fn ingest_galaxy_world_state(
    mut events: MessageReader<GameEvent>,
    mut cam: ResMut<GalaxyCameraPos>,
    mut warp: ResMut<WarpPhase>,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::GalaxyWorldState(gws) => {
                cam.pos_galaxy_units = gws.ship_position;
                warp.phase = gws.warp_phase;
            }
            NetEvent::Connected {
                galaxy_seed,
                system_seed,
                ..
            } => {
                if *galaxy_seed == 0 || *system_seed == 0 {
                    continue;
                }
                let map = voxeldust_core::galaxy::GalaxyMap::generate(*galaxy_seed);
                if let Some(star) = map.stars.iter().find(|s| s.system_seed == *system_seed) {
                    cam.pos_galaxy_units = star.position;
                    tracing::info!(
                        system_seed,
                        star_pos = ?(star.position.x, star.position.y, star.position.z),
                        "starfield: anchored on current system",
                    );
                }
            }
            _ => {}
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Mode crossfade
// ──────────────────────────────────────────────────────────────────────────

fn update_starfield_mode(
    time: Res<Time>,
    warp: Res<WarpPhase>,
    mut mode: ResMut<StarfieldMode>,
) {
    mode.target = match warp.phase {
        WARP_PHASE_CRUISE | WARP_PHASE_DECELERATE => 1.0,
        _ => 0.0,
    };
    let step = time.delta_secs() / TRANSITION_SECONDS;
    let delta = mode.target - mode.parallax_amount;
    if delta.abs() <= step {
        mode.parallax_amount = mode.target;
    } else {
        mode.parallax_amount += step * delta.signum();
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Mesh build / re-anchor
// ──────────────────────────────────────────────────────────────────────────

/// Rebuild the starfield mesh when:
///   * the catalog changed (new galaxy_seed or first ingest), or
///   * the camera has drifted past `REANCHOR_THRESHOLD_M` from the
///     current mesh anchor (only meaningful in / near parallax mode).
///
/// Mesh build cost is ~1 ms for 50k stars and happens at most a few
/// times per warp segment.
fn rebuild_starfield_mesh(
    mut commands: Commands,
    catalog: Res<StarCatalog>,
    gal_cam: Res<GalaxyCameraPos>,
    mut handles: ResMut<StarfieldHandles>,
    existing: Query<Entity, With<StarfieldEntity>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StarfieldMaterial>>,
) {
    if catalog.stars.is_empty() {
        // Drop any stale entity / asset handles when the catalog
        // empties (rare — full disconnect).
        if handles.mesh.is_some() || handles.material.is_some() {
            for e in &existing {
                commands.entity(e).despawn();
            }
            handles.mesh = None;
            handles.material = None;
            handles.built_for_seed = 0;
            handles.star_count = 0;
        }
        return;
    }

    // GalaxyCameraPos is in galaxy units (1 GU = 1e6 m). Convert to
    // metres for the threshold check — REANCHOR_THRESHOLD_M is metric.
    let drift_m =
        ((gal_cam.pos_galaxy_units - handles.anchor_galaxy_units).length() * GALAXY_UNIT_IN_BLOCKS)
            as f32;
    let needs_rebuild = handles.built_for_seed != catalog.galaxy_seed
        || handles.mesh.is_none()
        || drift_m > REANCHOR_THRESHOLD_M;
    if !needs_rebuild {
        return;
    }

    let new_anchor = gal_cam.pos_galaxy_units;
    let (mesh, star_count) = build_starfield_mesh(&catalog, new_anchor);
    let mesh_handle = meshes.add(mesh);

    // Lazy-init material on first build; reuse otherwise so the
    // existing entity's MeshMaterial3d handle stays valid.
    let material_handle = handles.material.clone().unwrap_or_else(|| {
        materials.add(StarfieldMaterial {
            uniform: StarfieldUniform {
                cam_offset: Vec3::ZERO,
                parallax_amount: 0.0,
                skybox_radius: SKYBOX_RADIUS_M,
            },
        })
    });

    // Despawn any previous starfield entity and respawn with the new
    // mesh handle. Single entity, so this is one despawn + one spawn.
    for e in &existing {
        commands.entity(e).despawn();
    }
    commands.spawn((
        StarfieldEntity,
        Mesh3d(mesh_handle.clone()),
        MeshMaterial3d(material_handle.clone()),
        // Camera sits at Bevy origin (floating-origin convention);
        // mesh positions are camera-relative once the shader subtracts
        // cam_offset, so the entity's Transform stays identity.
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
        Visibility::Visible,
        InheritedVisibility::default(),
        ViewVisibility::default(),
        Name::new("starfield"),
    ));

    handles.mesh = Some(mesh_handle);
    handles.material = Some(material_handle);
    handles.anchor_galaxy_units = new_anchor;
    handles.built_for_seed = catalog.galaxy_seed;
    handles.star_count = star_count;

    tracing::info!(
        galaxy_seed = catalog.galaxy_seed,
        stars = star_count,
        anchor_x = new_anchor.x,
        anchor_y = new_anchor.y,
        anchor_z = new_anchor.z,
        drift_m,
        "starfield: mesh rebuilt",
    );
}

/// Build the single starfield mesh: 4 verts + 6 indices per visible
/// star, attributes laid out for `STARFIELD_PARAMS_ATTR` plus the
/// standard POSITION + COLOR.
fn build_starfield_mesh(catalog: &StarCatalog, anchor: DVec3) -> (Mesh, u32) {
    let max_stars = catalog.stars.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(max_stars * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(max_stars * 4);
    let mut params: Vec<[f32; 4]> = Vec::with_capacity(max_stars * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(max_stars * 6);

    let corners = [(-1.0_f32, -1.0_f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let mut star_count: u32 = 0;

    for star in &catalog.stars {
        if star.luminosity < MIN_VISIBLE_LUMINOSITY {
            continue;
        }
        // star.position and anchor are in galaxy units (DVec3). The
        // shader expects metres so its skybox / parallax constants
        // (1×10⁶, 1.5×10⁶) make sense. Subtract in f64, scale to m,
        // *then* cast — keeps precision for the ~1e10 m absolute
        // values while landing well inside f32 safe range for the
        // anchor-relative differences (≤ REANCHOR_THRESHOLD_M).
        let rel_m = (star.position - anchor) * GALAXY_UNIT_IN_BLOCKS;
        let pos = [rel_m.x as f32, rel_m.y as f32, rel_m.z as f32];
        let (color_rgb, brightness) = star_color_and_brightness(star.star_class, star.luminosity);
        let color = [color_rgb[0], color_rgb[1], color_rgb[2], brightness];
        let half_size = STAR_BASE_HALF_SIZE_M * star.luminosity.cbrt().max(0.1);

        let base_idx = star_count * 4;
        for (cx, cy) in corners {
            positions.push(pos);
            colors.push(color);
            params.push([cx, cy, half_size, 0.0]);
        }
        // Two triangles per quad: (0,1,2) + (0,2,3).
        indices.extend_from_slice(&[
            base_idx,
            base_idx + 1,
            base_idx + 2,
            base_idx,
            base_idx + 2,
            base_idx + 3,
        ]);
        star_count += 1;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(STARFIELD_PARAMS_ATTR, params);
    mesh.insert_indices(Indices::U32(indices));
    (mesh, star_count)
}

/// Per-class colour + LDR brightness scalar.
///
/// HDR / Bloom / AgX are currently disabled (`main.rs:191-193`), so
/// the tonemap-aggressive emissive multipliers from the previous path
/// (140-1500) would all clamp to white. We keep brightness in [0,1]
/// here so stars read as a wide range of intensities even in plain
/// LDR. When HDR comes back, scale the `class_max` table up.
///
/// Class ordering (matches `StarClass` in `core::galaxy`):
///   0 = O (blue, hottest), 6 = M (red dwarf, dimmest).
fn star_color_and_brightness(class: u8, luminosity: f32) -> ([f32; 3], f32) {
    let (r, g, b, class_max) = match class {
        0 => (0.60, 0.72, 1.00, 1.00), // O — hot blue, brightest
        1 => (0.75, 0.85, 1.00, 0.85), // B
        2 => (0.92, 0.94, 1.00, 0.70), // A
        3 => (1.00, 0.98, 0.92, 0.55), // F
        4 => (1.00, 0.95, 0.80, 0.50), // G — sun-like
        5 => (1.00, 0.78, 0.55, 0.40), // K
        6 => (1.00, 0.55, 0.35, 0.30), // M — red dwarf
        _ => (1.00, 0.95, 0.85, 0.45),
    };
    let brightness = (class_max * luminosity.cbrt()).clamp(0.05, 1.0);
    ([r, g, b], brightness)
}

// ──────────────────────────────────────────────────────────────────────────
// Per-frame uniform update
// ──────────────────────────────────────────────────────────────────────────

/// Push the latest `cam_offset` + `parallax_amount` into the live
/// material's uniform. Runs every frame; cheap (one f32×4 write).
fn update_starfield_uniform(
    handles: Res<StarfieldHandles>,
    gal_cam: Res<GalaxyCameraPos>,
    _camera_world: Res<CameraWorldPos>,
    mode: Res<StarfieldMode>,
    mut materials: ResMut<Assets<StarfieldMaterial>>,
) {
    let Some(mat_handle) = handles.material.as_ref() else {
        return;
    };
    let Some(mat) = materials.get_mut(mat_handle) else {
        return;
    };
    // GU → metres so cam_offset shares units with star positions
    // baked into the mesh.
    let offset_m = (gal_cam.pos_galaxy_units - handles.anchor_galaxy_units) * GALAXY_UNIT_IN_BLOCKS;
    mat.uniform.cam_offset =
        Vec3::new(offset_m.x as f32, offset_m.y as f32, offset_m.z as f32);
    mat.uniform.parallax_amount = mode.parallax_amount;
    mat.uniform.skybox_radius = SKYBOX_RADIUS_M;
}
