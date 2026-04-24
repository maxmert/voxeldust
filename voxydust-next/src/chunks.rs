//! Voxel chunk rendering — Phase 5 of the Bevy migration plan.
//!
//! Generates voxel chunks using `voxeldust_core` and converts the resulting
//! quad list into Bevy `Mesh` assets. Chunks spawn as regular 3D entities with
//! `Mesh3d` + `MeshMaterial3d<StandardMaterial>`, so they pick up all of
//! Bevy's lighting, shadows, atmosphere, IBL, and post-processing for free.
//!
//! This is the "terrain layer" voxydust-next uses until we plumb the network
//! chunk stream (Phase 3) — a deterministic procedural test landscape built
//! from the same `chunk_mesher` the legacy voxydust renders with, so what we
//! see here is already 1:1 with the real game's meshing output.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use noise::{NoiseFn, SuperSimplex};
use voxeldust_core::block::{
    block_id::BlockId,
    chunk_mesher::{mesh_chunk, ChunkQuads, FACE_NORMALS},
    chunk_storage::ChunkStorage,
    palette::CHUNK_SIZE,
    registry::BlockRegistry,
};

/// Side length of one chunk in metres. Matches `voxeldust_core::CHUNK_SIZE`.
pub const CHUNK_SIDE: f32 = CHUNK_SIZE as f32;

/// Plugin that spawns a deterministic procedural terrain chunk grid at startup.
pub struct TerrainDemoPlugin;

impl Plugin for TerrainDemoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_demo_terrain);
    }
}

/// Spawn an N×N grid of chunks around the origin. Each chunk renders as a
/// single `Mesh` entity; Bevy's frustum culling handles out-of-view chunks,
/// and atmosphere/IBL/shadows apply automatically.
fn spawn_demo_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Single shared material: vertex colours carry per-block tint, PBR params
    // are globally tuned for "natural terrain" roughness. StandardMaterial
    // interacts with atmosphere IBL via the camera's
    // `AtmosphereEnvironmentMapLight`.
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.85,
        metallic: 0.0,
        // Use vertex colour as the surface colour.
        ..default()
    });

    // Single registry shared across chunk generation + meshing. The registry
    // is identical on server and client (`BlockRegistry::new()` is
    // deterministic).
    let registry = BlockRegistry::new();

    // 9×9 chunks = 558 × 558 m footprint, centred on origin. A wider grid
    // sells the horizon aerial-perspective effect better than the 7×7
    // demo grid — distant chunks visibly desaturate toward the sky colour.
    let half: i32 = 4;
    let noise = SuperSimplex::new(0xbeef_u32);
    let biome_noise = SuperSimplex::new(0xcafe_u32);

    for cx in -half..=half {
        for cz in -half..=half {
            // Generate the chunk contents — a deterministic heightmap driven
            // by multi-octave simplex noise. Sampled at 1-block resolution in
            // world space so adjacent chunks stitch seamlessly.
            let chunk = build_terrain_chunk(&noise, &biome_noise, cx, cz);

            // Mesh it. No neighbours yet (Phase 3 brings streaming) so
            // chunk-boundary quads will render — gives us a handy visual
            // indicator of chunk edges until neighbour-meshing is in.
            let neighbours: [Option<&ChunkStorage>; 6] = [None; 6];
            let quads = mesh_chunk(&chunk, &neighbours, &registry, false);

            if quads.quads.is_empty() {
                continue;
            }

            let mesh = quads_to_bevy_mesh(&quads, &registry);

            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.clone()),
                // Chunks are positioned in world space at chunk_index × CHUNK_SIDE.
                // Scene units are metres — 1 Bevy unit = 1 metre.
                Transform::from_xyz(
                    cx as f32 * CHUNK_SIDE,
                    0.0,
                    cz as f32 * CHUNK_SIDE,
                ),
                Name::new(format!("chunk_{cx}_{cz}")),
            ));
        }
    }
}

/// Build a single chunk's block contents from a world-space heightmap. Three
/// frequency bands drive elevation (continental, rolling, detail) and a
/// separate biome field selects surface block + roughness. Result: plains,
/// hills, mountains, and beaches within the same contiguous world.
///
/// Water is below y = 8; beaches sit right at the waterline; grass covers
/// mid-elevations; stone exposes on peaks.
fn build_terrain_chunk(
    noise: &SuperSimplex,
    biome_noise: &SuperSimplex,
    cx: i32,
    cz: i32,
) -> ChunkStorage {
    const WATER_LEVEL: u8 = 8;
    const BEACH_TOP: u8 = 10;
    const ROCKY_FLOOR: u8 = 40;

    let mut chunk = ChunkStorage::new_empty();
    let chunk_x0 = cx * CHUNK_SIZE as i32;
    let chunk_z0 = cz * CHUNK_SIZE as i32;

    for bx in 0..CHUNK_SIZE as u8 {
        for bz in 0..CHUNK_SIZE as u8 {
            let wx = chunk_x0 + bx as i32;
            let wz = chunk_z0 + bz as i32;

            // Continental: slow 400-m-wavelength ridges — picks the broad
            // "is this a mountain ridge or a valley?" answer.
            let continent = noise.get([wx as f64 / 400.0, wz as f64 / 400.0, 0.0]);
            // Rolling: 140-m hills on top of continents.
            let rolling = noise.get([wx as f64 / 140.0, wz as f64 / 140.0, 1.0]);
            // Detail: ±2 m bumps at 22-m wavelength.
            let detail = noise.get([wx as f64 / 22.0, wz as f64 / 22.0, 11.0]);

            // Bias toward positive so water appears in natural lowlands only.
            let height_f =
                16.0 + continent * 22.0 + rolling * 10.0 + detail * 2.0;
            let height = height_f.clamp(1.0, (CHUNK_SIZE - 2) as f64) as u8;

            // Biome selector — independent noise so biome boundaries don't
            // correlate with elevation features, producing visual variety.
            let biome = biome_noise.get([wx as f64 / 260.0, wz as f64 / 260.0, 0.0]);

            for by in 0..height {
                let id = if by < WATER_LEVEL.saturating_sub(1) && by < height.saturating_sub(3) {
                    // Deep: mostly stone with sand pockets lining lake beds.
                    BlockId::STONE
                } else if by <= BEACH_TOP && by >= height.saturating_sub(2) {
                    // Shoreline: sand above water, up to beach top.
                    BlockId::SAND
                } else if by >= ROCKY_FLOOR && by >= height.saturating_sub(1) {
                    // High peaks: bare stone.
                    BlockId::STONE
                } else if by >= height.saturating_sub(1) {
                    // Surface layer — biome-selected.
                    if biome > 0.35 {
                        BlockId::SAND
                    } else if biome < -0.35 {
                        BlockId::DIRT
                    } else {
                        BlockId::GRASS
                    }
                } else if by >= height.saturating_sub(4) {
                    BlockId::DIRT
                } else {
                    BlockId::STONE
                };
                chunk.set_block(bx, by, bz, id);
            }
        }
    }
    chunk
}

/// Convert voxeldust-core's greedy-meshed `ChunkQuads` into a Bevy `Mesh`.
/// Positions are chunk-local metres; we flip the winding so Bevy's default
/// front-face (CCW) renders the outward-facing side.
fn quads_to_bevy_mesh(quads: &ChunkQuads, registry: &BlockRegistry) -> Mesh {
    let count = quads.quads.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(count * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(count * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(count * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(count * 6);

    for quad in &quads.quads {
        let normal = FACE_NORMALS[quad.face as usize];
        let def = registry.get(BlockId::from_u16(quad.block_id));
        let color = [
            def.color_hint[0] as f32 / 255.0,
            def.color_hint[1] as f32 / 255.0,
            def.color_hint[2] as f32 / 255.0,
            1.0,
        ];

        let base = positions.len() as u32;
        for vert in &quad.vertices {
            positions.push([vert[0] as f32, vert[1] as f32, vert[2] as f32]);
            normals.push(normal);
            colors.push(color);
        }

        // Winding order: legacy voxydust used (2, 0, 1, 1, 3, 2) against a
        // CCW front face with `FrontFace::Ccw + CullMode::Back`. Bevy's
        // `Mesh` with default material uses the same convention, so pass
        // the same indices verbatim.
        indices.extend_from_slice(&[base + 2, base, base + 1, base + 1, base + 3, base + 2]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        VertexAttributeValues::Float32x4(colors),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
