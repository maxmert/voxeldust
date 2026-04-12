//! Block mesh GPU rendering — vertex format, pipeline, per-chunk buffers, draw management.
//!
//! ## Multi-Source Architecture
//!
//! The renderer supports chunks from multiple shard connections simultaneously.
//! Each chunk source (ship shard, planet shard, etc.) has its own set of GPU
//! buffers. This enables:
//! - Ship interior visible while walking on a planet (ship chunks persist)
//! - Planet terrain visible while inside a ship (planet chunks alongside ship chunks)
//! - Seamless shard transitions without visual pop-in/pop-out
//!
//! GPU mesh buffers are keyed by `ChunkKey = (ChunkSourceId, IVec3)` —
//! no collisions between sources that share the same chunk coordinates.
//!
//! The render pass groups chunks by source and applies source-specific transforms:
//! - Ship sources: flat Cartesian with ship rotation + floating-origin offset
//! - Planet sources (future): sphere-projected with planet transform

use std::collections::HashMap;

use glam::{DVec3, IVec3, Mat4, Vec3};
use wgpu::util::DeviceExt;

use voxeldust_core::block::chunk_mesher::{self, ChunkQuads, MesherQuad, FACE_NORMALS};
use voxeldust_core::block::client_chunks::{ChunkKey, ChunkSourceId};
use voxeldust_core::block::sub_block_mesher::{self, SubBlockMeshData};
use voxeldust_core::block::{BlockId, BlockRegistry, ChunkStorage};

// ---------------------------------------------------------------------------
// BlockVertex — GPU vertex format (client-only)
// ---------------------------------------------------------------------------

/// A single vertex for block mesh rendering.
///
/// 36 bytes per vertex — position (12), normal (12), color (12).
///
/// **Coordinate precision strategy:**
/// - `position` is in **chunk-local space** (0..62 range), so f32 is always exact.
/// - Large-scale positioning (planet radius 100K+ blocks) is handled by the
///   per-chunk model matrix uniform, computed on CPU as:
///   `chunk_world_position_f64 - camera_position_f64` → f32 (camera-relative, fits in f32).
/// - This is the floating-origin pattern — same as celestial body rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

unsafe impl bytemuck::Pod for BlockVertex {}
unsafe impl bytemuck::Zeroable for BlockVertex {}

impl BlockVertex {
    pub const SIZE: usize = std::mem::size_of::<Self>(); // 36
    pub const POSITION_OFFSET: u64 = 0;
    pub const NORMAL_OFFSET: u64 = 12;
    pub const COLOR_OFFSET: u64 = 24;
}

// ---------------------------------------------------------------------------
// Quad → GPU vertex conversion
// ---------------------------------------------------------------------------

/// GPU-ready mesh data for a single chunk.
pub struct ChunkMeshData {
    pub vertices: Vec<BlockVertex>,
    pub indices: Vec<u32>,
}

impl ChunkMeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Convert format-agnostic `ChunkQuads` to GPU vertex data.
pub fn quads_to_mesh(quads: &ChunkQuads, registry: &BlockRegistry) -> ChunkMeshData {
    if quads.is_empty() {
        return ChunkMeshData {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let count = quads.quad_count();
    let mut vertices = Vec::with_capacity(count * 4);
    let mut indices = Vec::with_capacity(count * 6);

    for quad in &quads.quads {
        let normal = FACE_NORMALS[quad.face as usize];
        let color = block_color(quad.block_id, registry);

        let base = vertices.len() as u32;
        for vert in &quad.vertices {
            vertices.push(BlockVertex {
                position: [vert[0] as f32, vert[1] as f32, vert[2] as f32],  // i8 → f32
                normal,
                color,
            });
        }

        // Two triangles per quad: 2-0-1 and 1-3-2 (bgm's winding order).
        indices.push(base + 2);
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 1);
        indices.push(base + 3);
        indices.push(base + 2);
    }

    ChunkMeshData { vertices, indices }
}

/// Generate GPU mesh data directly from a ChunkStorage (convenience wrapper).
pub fn generate_chunk_gpu_mesh(
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
) -> ChunkMeshData {
    let quads = chunk_mesher::mesh_chunk(chunk, neighbors, registry, transparent_pass);
    quads_to_mesh(&quads, registry)
}

/// Generate a GPU mesh for a chunk filtered by sub-grid membership.
///
/// - `sub_grid_filter = None`: mesh only root-grid blocks.
/// - `sub_grid_filter = Some(id)`: mesh only blocks in the specified sub-grid.
pub fn generate_chunk_gpu_mesh_filtered(
    chunk: &ChunkStorage,
    chunk_key: glam::IVec3,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
    sub_grid_assignments: &std::collections::HashMap<glam::IVec3, u32>,
    sub_grid_filter: Option<u32>,
) -> ChunkMeshData {
    let quads = chunk_mesher::mesh_chunk_filtered(
        chunk, chunk_key, neighbors, registry, transparent_pass,
        sub_grid_assignments, sub_grid_filter,
    );
    quads_to_mesh(&quads, registry)
}

/// Like `generate_chunk_gpu_mesh`, but reuses a padded voxel buffer to avoid 512KB
/// allocation per mesh call. Use when meshing many chunks in a loop.
pub fn generate_chunk_gpu_mesh_with_buf(
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
    voxel_buf: &mut Vec<u16>,
) -> ChunkMeshData {
    let quads = chunk_mesher::mesh_chunk_with_buf(chunk, neighbors, registry, transparent_pass, voxel_buf);
    quads_to_mesh(&quads, registry)
}

/// Like `generate_chunk_gpu_mesh_filtered`, but reuses a padded voxel buffer.
pub fn generate_chunk_gpu_mesh_filtered_with_buf(
    chunk: &ChunkStorage,
    chunk_key: glam::IVec3,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
    sub_grid_assignments: &std::collections::HashMap<glam::IVec3, u32>,
    sub_grid_filter: Option<u32>,
    voxel_buf: &mut Vec<u16>,
) -> ChunkMeshData {
    let quads = chunk_mesher::mesh_chunk_filtered_with_buf(
        chunk, chunk_key, neighbors, registry, transparent_pass,
        sub_grid_assignments, sub_grid_filter, voxel_buf,
    );
    quads_to_mesh(&quads, registry)
}

#[inline]
fn block_color(block_id: u16, registry: &BlockRegistry) -> [f32; 3] {
    let def = registry.get(BlockId::from_u16(block_id));
    [
        def.color_hint[0] as f32 / 255.0,
        def.color_hint[1] as f32 / 255.0,
        def.color_hint[2] as f32 / 255.0,
    ]
}

// ---------------------------------------------------------------------------
// GPU buffer management (multi-source)
// ---------------------------------------------------------------------------

/// GPU buffers for a single chunk's block mesh.
pub struct ChunkGpuMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
}

/// Key for a sub-grid chunk mesh: identifies a specific sub-grid's blocks within a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubGridMeshKey {
    pub source: ChunkSourceId,
    pub chunk: IVec3,
    pub sub_grid_id: u32,
}

/// Manages all block-related GPU state: pipelines + per-source per-chunk mesh buffers.
///
/// Chunks are keyed by `ChunkKey = (ChunkSourceId, IVec3)` so multiple sources
/// can have overlapping chunk positions without collision.
pub struct BlockRenderer {
    /// Render pipeline for opaque block meshes (backface culling, PBR lighting).
    pub pipeline: wgpu::RenderPipeline,
    /// Shadow depth-only pipeline for block meshes (CSM cascade rendering).
    pub shadow_pipeline: wgpu::RenderPipeline,
    /// Per-source, per-chunk GPU mesh buffers for full blocks (root grid only).
    chunk_meshes: HashMap<ChunkKey, ChunkGpuMesh>,
    /// Per-source, per-chunk GPU mesh buffers for sub-block elements.
    /// Uses the same vertex format and pipeline as full blocks.
    sub_block_meshes: HashMap<ChunkKey, ChunkGpuMesh>,
    /// Per-sub-grid, per-chunk GPU mesh buffers for mechanical sub-grid blocks.
    /// These are drawn with per-sub-grid transforms (rotation/translation from server).
    sub_grid_meshes: HashMap<SubGridMeshKey, ChunkGpuMesh>,
}

impl BlockRenderer {
    /// Create the block renderer with its pipelines.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        bind_group_layout: &wgpu::BindGroupLayout,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        voxel_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let block_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("block.wgsl").into()),
        });

        let depth_format = wgpu::TextureFormat::Depth32Float;

        let block_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: BlockVertex::SIZE as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: BlockVertex::POSITION_OFFSET,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: BlockVertex::NORMAL_OFFSET,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: BlockVertex::COLOR_OFFSET,
                    shader_location: 2,
                },
            ],
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("block_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout, scene_bind_group_layout, voxel_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("block_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &block_shader,
                entry_point: Some("vs_main"),
                buffers: &[block_vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &block_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Block shadow pipeline: depth-only rendering for CSM cascade layers.
        let block_shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("block_shadow_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout], // only group 0
            push_constant_ranges: &[],
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("block_shadow_pipeline"),
            layout: Some(&block_shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &block_shader,
                entry_point: Some("vs_main"),
                buffers: &[block_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: None, // depth-only
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            shadow_pipeline,
            chunk_meshes: HashMap::new(),
            sub_block_meshes: HashMap::new(),
            sub_grid_meshes: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // GPU buffer lifecycle (source-aware)
    // -----------------------------------------------------------------------

    /// Upload a chunk's mesh data to the GPU. Keyed by `(source, chunk_pos)`.
    pub fn upload_chunk_mesh(
        &mut self,
        device: &wgpu::Device,
        key: ChunkKey,
        mesh: &ChunkMeshData,
    ) {
        if mesh.is_empty() {
            self.chunk_meshes.remove(&key);
            return;
        }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("block_vb_s{}_c{}", key.source.0, key.chunk)),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("block_ib_s{}_c{}", key.source.0, key.chunk)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.chunk_meshes.insert(key, ChunkGpuMesh {
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
        });
    }

    /// Remove a specific chunk's GPU buffers.
    pub fn remove_chunk_mesh(&mut self, key: ChunkKey) {
        self.chunk_meshes.remove(&key);
    }

    /// Remove ALL GPU buffers belonging to a source (when source is disconnected).
    pub fn remove_source(&mut self, source: ChunkSourceId) {
        self.chunk_meshes.retain(|k, _| k.source != source);
        self.sub_block_meshes.retain(|k, _| k.source != source);
        self.sub_grid_meshes.retain(|k, _| k.source != source);
    }

    /// Total number of chunks with active GPU meshes across all sources.
    pub fn total_chunk_count(&self) -> usize {
        self.chunk_meshes.len()
    }

    /// Whether any source has chunks to render.
    pub fn has_chunks(&self) -> bool {
        !self.chunk_meshes.is_empty()
    }

    /// Iterate all chunks belonging to a specific source.
    pub fn chunks_for_source(&self, source: ChunkSourceId) -> impl Iterator<Item = (IVec3, &ChunkGpuMesh)> {
        self.chunk_meshes.iter()
            .filter(move |(k, _)| k.source == source)
            .map(|(k, m)| (k.chunk, m))
    }

    /// All unique source IDs that have at least one chunk.
    pub fn active_sources(&self) -> Vec<ChunkSourceId> {
        let mut sources: Vec<ChunkSourceId> = self.chunk_meshes.keys()
            .map(|k| k.source)
            .collect();
        sources.sort_unstable_by_key(|s| s.0);
        sources.dedup();
        sources
    }

    // -----------------------------------------------------------------------
    // Sub-block mesh management
    // -----------------------------------------------------------------------

    /// Upload sub-block mesh data to the GPU for a specific chunk.
    pub fn upload_sub_block_mesh(
        &mut self,
        device: &wgpu::Device,
        key: ChunkKey,
        mesh: &SubBlockMeshData,
    ) {
        if mesh.is_empty() {
            self.sub_block_meshes.remove(&key);
            return;
        }

        // SubBlockVertex has the same layout as BlockVertex (position, normal, color).
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("subblock_vb_s{}_c{}", key.source.0, key.chunk)),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("subblock_ib_s{}_c{}", key.source.0, key.chunk)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.sub_block_meshes.insert(key, ChunkGpuMesh {
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
        });
    }

    /// Remove sub-block GPU buffers for a specific chunk.
    pub fn remove_sub_block_mesh(&mut self, key: ChunkKey) {
        self.sub_block_meshes.remove(&key);
    }

    /// Iterate all sub-block meshes belonging to a specific source.
    pub fn sub_blocks_for_source(&self, source: ChunkSourceId) -> impl Iterator<Item = (IVec3, &ChunkGpuMesh)> {
        self.sub_block_meshes.iter()
            .filter(move |(k, _)| k.source == source)
            .map(|(k, m)| (k.chunk, m))
    }

    // -----------------------------------------------------------------------
    // Sub-grid mesh management (mechanical mounts: rotors, pistons)
    // -----------------------------------------------------------------------

    /// Upload a sub-grid chunk mesh to the GPU.
    pub fn upload_sub_grid_mesh(
        &mut self,
        device: &wgpu::Device,
        key: SubGridMeshKey,
        mesh: &ChunkMeshData,
    ) {
        if mesh.vertices.is_empty() {
            self.sub_grid_meshes.remove(&key);
            return;
        }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("sg{}_vb_s{}_c{}", key.sub_grid_id, key.source.0, key.chunk)),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("sg{}_ib_s{}_c{}", key.sub_grid_id, key.source.0, key.chunk)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.sub_grid_meshes.insert(key, ChunkGpuMesh {
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
        });
    }

    /// Remove all sub-grid meshes for a specific sub-grid ID (when sub-grid is destroyed).
    pub fn remove_sub_grid_meshes(&mut self, sub_grid_id: u32) {
        self.sub_grid_meshes.retain(|k, _| k.sub_grid_id != sub_grid_id);
    }

    /// Remove sub-grid meshes for a specific source (when source is disconnected).
    pub fn remove_sub_grid_source(&mut self, source: ChunkSourceId) {
        self.sub_grid_meshes.retain(|k, _| k.source != source);
    }

    /// Iterate all sub-grid meshes belonging to a specific source.
    pub fn sub_grid_meshes_for_source(&self, source: ChunkSourceId) -> impl Iterator<Item = (&SubGridMeshKey, &ChunkGpuMesh)> {
        self.sub_grid_meshes.iter()
            .filter(move |(k, _)| k.source == source)
    }

    // -----------------------------------------------------------------------
    // Transform computation
    // -----------------------------------------------------------------------

    /// Compute the model matrix for a chunk using floating-origin.
    ///
    /// `base_transform`: the source-specific transform that maps source-local
    /// coordinates to camera-relative world space. For a ship source, this is
    /// `Mat4::from_translation(ship_origin_offset) * ship_rot_mat`.
    /// For a planet source (future), it would include sphere projection.
    ///
    /// `chunk_key`: the IVec3 chunk position within the source grid.
    /// `chunk_size`: blocks per chunk axis (62).
    pub fn chunk_model_matrix(
        base_transform: Mat4,
        chunk_key: IVec3,
        chunk_size: f64,
    ) -> Mat4 {
        let chunk_offset = Vec3::new(
            (chunk_key.x as f64 * chunk_size) as f32,
            (chunk_key.y as f64 * chunk_size) as f32,
            (chunk_key.z as f64 * chunk_size) as f32,
        );
        base_transform * Mat4::from_translation(chunk_offset)
    }
}
