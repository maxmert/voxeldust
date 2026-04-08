//! GPU voxel volume — uploads nearby chunk block data to a 3D texture for
//! voxel ray-marched sun shadows and block light propagation.
//!
//! The volume is centred on the player's position and contains occupancy +
//! emission data from ALL active chunk sources (ship + planet), enabling
//! cross-shard lighting automatically.

use glam::IVec3;
use wgpu::util::DeviceExt;

use voxeldust_core::block::client_chunks::{ClientChunkCache, ChunkSourceId};
use voxeldust_core::block::palette::CHUNK_SIZE;
use voxeldust_core::block::BlockRegistry;

use crate::graphics_settings::GraphicsSettings;

/// Per-voxel data packed into the occupancy 3D texture (Rgba8Uint).
/// R = solid flag (0 = air/transparent, 255 = solid)
/// G = light_emission (0-255, maps to 0-15 block emission scaled up)
/// B = block type lower 8 bits (for emission color lookup)
/// A = reserved (0)
const BYTES_PER_VOXEL: usize = 4;

/// GPU-side uniform passed to shaders so they can convert world positions to
/// volume-index coordinates.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVolumeParams {
    /// Transforms fragment world_pos (camera-relative rotated) to volume-index
    /// coordinates (0..size range). Bakes in inverse(base_transform) and
    /// the volume_origin offset.
    pub world_to_volume: [[f32; 4]; 4],
    /// xyz = sun direction in volume space (for DDA ray march). w = volume size.
    pub sun_dir_and_size: [f32; 4],
    /// x = 1.0 / volume_size. yzw = pad.
    pub inv_volume_size: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<VoxelVolumeParams>() == 96);

/// Manages the GPU 3D textures for voxel lighting.
pub struct VoxelVolume {
    /// Current volume size (one axis). 64, 128, or 256.
    size: u32,

    /// 3D occupancy texture: Rgba8Uint, N³.
    pub occupancy_texture: wgpu::Texture,
    pub occupancy_view: wgpu::TextureView,

    /// Block light ping-pong textures: Rgba16Float, N³.
    pub light_texture_a: wgpu::Texture,
    pub light_view_a: wgpu::TextureView,
    pub light_texture_b: wgpu::Texture,
    pub light_view_b: wgpu::TextureView,

    /// Storage views for compute shader write access (same textures, storage binding).
    light_storage_view_a: wgpu::TextureView,
    light_storage_view_b: wgpu::TextureView,

    /// Trilinear sampler for reading the light texture in fragment shaders.
    pub light_sampler: wgpu::Sampler,

    /// Uniform buffer with volume origin / size.
    pub params_buf: wgpu::Buffer,

    /// Compute pipeline for light propagation.
    propagate_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for the propagation compute shader.
    propagate_bind_group_layout: wgpu::BindGroupLayout,
    /// Ping-pong bind groups: [0] reads A→writes B, [1] reads B→writes A.
    propagate_bind_groups: [wgpu::BindGroup; 2],

    /// CPU staging buffer for occupancy data (size³ × 4 bytes).
    staging: Vec<u8>,

    /// CPU zero buffer for clearing light_texture_a (size³ × 8 bytes for Rgba16Float).
    light_clear_buf: Vec<u8>,

    /// Volume origin in block coordinates (corner of the volume).
    origin: IVec3,

    /// Whether the volume needs to be re-uploaded to the GPU.
    pub needs_upload: bool,

    /// Whether the block light needs re-propagation.
    pub needs_propagation: bool,
}

impl VoxelVolume {
    /// Allocate GPU resources for the voxel volume.
    pub fn new(device: &wgpu::Device, settings: &GraphicsSettings) -> Self {
        let size = settings.voxel_volume_size;
        let (occupancy_texture, occupancy_view) = create_occupancy_texture(device, size);
        let (light_texture_a, light_view_a) = create_light_texture(device, size, "light_a");
        let (light_texture_b, light_view_b) = create_light_texture(device, size, "light_b");

        // Storage views for compute shader write access.
        let light_storage_view_a = light_texture_a.create_view(&wgpu::TextureViewDescriptor {
            label: Some("light_a_storage"),
            ..Default::default()
        });
        let light_storage_view_b = light_texture_b.create_view(&wgpu::TextureViewDescriptor {
            label: Some("light_b_storage"),
            ..Default::default()
        });

        let light_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("voxel_light_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_volume_params"),
            size: std::mem::size_of::<VoxelVolumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -- Compute pipeline for light propagation --
        let propagate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light_propagate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("light_propagate.wgsl").into()),
        });

        let propagate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("propagate_layout"),
                entries: &[
                    // binding 0: occupancy (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: light_prev (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: light_out (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    // binding 3: volume params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<VoxelVolumeParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let propagate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("propagate_pipeline_layout"),
                bind_group_layouts: &[&propagate_bind_group_layout],
                push_constant_ranges: &[],
            });

        let propagate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("propagate_pipeline"),
                layout: Some(&propagate_pipeline_layout),
                module: &propagate_shader,
                entry_point: Some("propagate"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Ping-pong bind groups:
        // [0]: read A, write B
        // [1]: read B, write A
        let propagate_bind_groups = [
            create_propagate_bind_group(
                device,
                &propagate_bind_group_layout,
                &occupancy_view,
                &light_view_a,
                &light_storage_view_b,
                &params_buf,
                "propagate_bg_a2b",
            ),
            create_propagate_bind_group(
                device,
                &propagate_bind_group_layout,
                &occupancy_view,
                &light_view_b,
                &light_storage_view_a,
                &params_buf,
                "propagate_bg_b2a",
            ),
        ];

        let staging = vec![0u8; (size as usize).pow(3) * BYTES_PER_VOXEL];
        // Rgba16Float = 8 bytes per texel. Only need to clear texture A.
        let light_clear_buf = vec![0u8; (size as usize).pow(3) * 8];

        Self {
            size,
            occupancy_texture,
            occupancy_view,
            light_texture_a,
            light_view_a,
            light_texture_b,
            light_view_b,
            light_storage_view_a,
            light_storage_view_b,
            light_sampler,
            params_buf,
            propagate_pipeline,
            propagate_bind_group_layout,
            propagate_bind_groups,
            staging,
            light_clear_buf,
            origin: IVec3::ZERO,
            needs_upload: false,
            needs_propagation: false,
        }
    }

    /// Recreate textures if the volume size setting has changed.
    /// Returns true if the volume was resized (caller must recreate bind groups).
    pub fn resize_if_needed(&mut self, device: &wgpu::Device, settings: &GraphicsSettings) -> bool {
        if settings.voxel_volume_size != self.size {
            *self = Self::new(device, settings);
            true
        } else {
            false
        }
    }

    /// Current volume size (one axis).
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Populate the staging buffer from nearby chunk data. Call once per frame
    /// (or when dirty). Returns true if any data changed.
    ///
    /// `player_block_pos` is the player's position in block/world coordinates
    /// (integer, same coordinate space as chunk_pos * CHUNK_SIZE + block_xyz).
    pub fn populate(
        &mut self,
        player_block_pos: IVec3,
        cache: &ClientChunkCache,
        registry: &BlockRegistry,
        active_sources: &[ChunkSourceId],
    ) -> bool {
        let half = (self.size / 2) as i32;
        let new_origin = player_block_pos - IVec3::splat(half);

        // Skip if player hasn't moved (origin unchanged) and no external dirty flag.
        if new_origin == self.origin && !self.needs_upload {
            return false;
        }

        self.origin = new_origin;
        let sz = self.size as i32;

        // Clear staging buffer.
        self.staging.fill(0);

        let chunk_sz = CHUNK_SIZE as i32;

        for &source in active_sources {
            for chunk_pos in cache.source_chunk_keys(source) {
                let chunk = match cache.get_chunk(source, chunk_pos) {
                    Some(c) => c,
                    None => continue,
                };

                // Chunk covers blocks [chunk_pos * 62, chunk_pos * 62 + 61].
                let chunk_block_min = chunk_pos * chunk_sz;
                let chunk_block_max = chunk_block_min + IVec3::splat(chunk_sz);

                // Volume covers blocks [origin, origin + size).
                let vol_min = self.origin;
                let vol_max = self.origin + IVec3::splat(sz);

                // Overlap region.
                let overlap_min = chunk_block_min.max(vol_min);
                let overlap_max = chunk_block_max.min(vol_max);

                if overlap_min.x >= overlap_max.x
                    || overlap_min.y >= overlap_max.y
                    || overlap_min.z >= overlap_max.z
                {
                    continue; // No overlap.
                }

                // Iterate blocks in the overlap.
                for bx in overlap_min.x..overlap_max.x {
                    for by in overlap_min.y..overlap_max.y {
                        for bz in overlap_min.z..overlap_max.z {
                            // Chunk-local coordinates.
                            let local_x = (bx - chunk_block_min.x) as u8;
                            let local_y = (by - chunk_block_min.y) as u8;
                            let local_z = (bz - chunk_block_min.z) as u8;

                            let block_id = chunk.get_block(local_x, local_y, local_z);
                            if block_id.is_air() {
                                continue;
                            }

                            let def = registry.get(block_id);

                            // Volume-local coordinates.
                            let vx = (bx - self.origin.x) as usize;
                            let vy = (by - self.origin.y) as usize;
                            let vz = (bz - self.origin.z) as usize;

                            // Z-major layout matching wgpu 3D texture memory:
                            // X varies fastest (within a row), Y next (rows per slice), Z outermost (slices).
                            let idx = (vz * (sz as usize) * (sz as usize)
                                + vy * (sz as usize)
                                + vx)
                                * BYTES_PER_VOXEL;

                            if idx + BYTES_PER_VOXEL > self.staging.len() {
                                continue;
                            }

                            // R: light-blocking flag. Solid AND opaque blocks block light.
                            // Transparent blocks (windows, glass) let light pass through.
                            self.staging[idx] = if def.is_solid && !def.is_transparent { 255 } else { 0 };
                            // G: light emission (0-15 scaled to 0-255 for GPU precision).
                            self.staging[idx + 1] = (def.light_emission as u16 * 17).min(255) as u8;
                            // B: block type lower 8 bits.
                            self.staging[idx + 2] = (block_id.as_u16() & 0xFF) as u8;
                            // A: reserved.
                            self.staging[idx + 3] = 0;
                        }
                    }
                }
            }
        }

        self.needs_upload = true;
        self.needs_propagation = true;
        true
    }

    /// Upload the staging buffer to the GPU occupancy texture and update the
    /// params uniform.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if !self.needs_upload {
            return;
        }
        self.needs_upload = false;

        // Upload occupancy texture.
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.occupancy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.staging,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.size * BYTES_PER_VOXEL as u32),
                rows_per_image: Some(self.size),
            },
            wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: self.size,
            },
        );

    }

    /// Compute the volume params for the current frame. Must be called every frame
    /// (not just when volume changes) because `base_transform` changes with player movement.
    ///
    /// `base_transform` is the same transform used for chunk rendering (e.g. the interior
    /// ship transform: `from_translation(ship_rot * -(player_pos + eye)) * ship_rot`).
    /// `sun_direction_world` is the sun direction in the same space as fragment `world_pos`.
    pub fn compute_params(
        &self,
        base_transform: glam::Mat4,
        sun_direction_world: glam::Vec3,
    ) -> VoxelVolumeParams {
        // base_transform maps ship-local block coords to camera-relative world space:
        //   world_pos = base_transform * ship_local_pos
        //
        // We need the inverse: ship_local_pos = inverse(base_transform) * world_pos
        // Then: volume_index = ship_local_pos - volume_origin
        //
        // Combined: world_to_volume = translate(-volume_origin) * inverse(base_transform)
        let inv_base = base_transform.inverse();
        let vol_origin_f32 = glam::Vec3::new(
            self.origin.x as f32,
            self.origin.y as f32,
            self.origin.z as f32,
        );
        let world_to_volume = glam::Mat4::from_translation(-vol_origin_f32) * inv_base;

        // Transform sun direction to volume space (rotation only, no translation).
        let sun_vol = inv_base.transform_vector3(sun_direction_world).normalize();

        VoxelVolumeParams {
            world_to_volume: world_to_volume.to_cols_array_2d(),
            sun_dir_and_size: [sun_vol.x, sun_vol.y, sun_vol.z, self.size as f32],
            inv_volume_size: [1.0 / self.size as f32, 0.0, 0.0, 0.0],
        }
    }

    /// Zero-fill light_texture_a before propagation to prevent uninitialized data
    /// from flooding the volume. Only texture A needs clearing — texture B is fully
    /// overwritten in iteration 0 by the compute shader.
    pub fn clear_light_texture_a(&self, queue: &wgpu::Queue) {
        let extent = wgpu::Extent3d {
            width: self.size,
            height: self.size,
            depth_or_array_layers: self.size,
        };
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.light_texture_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.light_clear_buf,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.size * 8), // Rgba16Float = 8 bytes/texel
                rows_per_image: Some(self.size),
            },
            extent,
        );
    }

    /// Run the block light propagation compute shader. Call after upload() when
    /// needs_propagation is true. Dispatches `iterations` ping-pong passes.
    ///
    /// The encoder is passed in so this can be batched with the render passes
    /// in a single command buffer submission.
    pub fn propagate_light(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
    ) {
        if !self.needs_propagation {
            return;
        }
        self.needs_propagation = false;

        let workgroups = (self.size + 3) / 4; // ceil(size / 4)

        for i in 0..iterations {
            // Alternate bind groups: even iterations read A→write B, odd read B→write A.
            let bg_idx = (i % 2) as usize;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("light_propagate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.propagate_pipeline);
            pass.set_bind_group(0, &self.propagate_bind_groups[bg_idx], &[]);
            pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        // After an odd number of iterations the result is in texture B.
        // After an even number, it's in texture A. The fragment shader always
        // reads light_view_a (bound in the voxel bind group). If the result
        // ended up in B, we need one extra copy pass — OR we can just ensure
        // we always do an even number of iterations so the result lands in A.
        // The GraphicsSettings presets use 8, 12, 15 — we round up to even.
        // (The caller should pass an even iteration count.)
    }

    /// Whether the last propagation result is in texture A (which is what the
    /// fragment shader reads). Returns true if iterations was even.
    pub fn result_in_a(iterations: u32) -> bool {
        iterations % 2 == 0
    }
}

// ---------------------------------------------------------------------------
// GPU texture helpers
// ---------------------------------------------------------------------------

fn create_occupancy_texture(
    device: &wgpu::Device,
    size: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel_occupancy"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_light_texture(
    device: &wgpu::Device,
    size: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_propagate_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    occupancy_view: &wgpu::TextureView,
    light_read_view: &wgpu::TextureView,
    light_write_view: &wgpu::TextureView,
    params_buf: &wgpu::Buffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(occupancy_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(light_read_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(light_write_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    })
}
