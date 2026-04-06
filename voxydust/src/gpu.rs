//! GPU state: device, surface, pipelines, buffers, and GPU-related types.

use std::sync::Arc;

use tracing::info;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::mesh::IcoSphere;
use crate::stars::{StarInstance, StarSceneUniforms, MAX_STAR_INSTANCES};

pub const MAX_OBJECTS: usize = 2048;

/// Eye height offset from player position. Player capsule center is at ~1.0m,
/// so eye is at capsule center + 0.5m (roughly head height of 1.5m total).
pub const EYE_HEIGHT: f64 = 0.5;

/// Six frustum planes extracted from a view-projection matrix (Gribb-Hartmann method).
/// Each plane is (a, b, c, d) where ax + by + cz + d >= 0 is inside.
pub struct FrustumPlanes {
    pub planes: [[f32; 4]; 6],
}

impl FrustumPlanes {
    pub fn from_vp(vp: &glam::Mat4) -> Self {
        let r = vp.to_cols_array_2d();
        // Extract rows from the column-major matrix.
        let row = |i: usize| -> [f32; 4] {
            [r[0][i], r[1][i], r[2][i], r[3][i]]
        };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let mut planes = [[0.0f32; 4]; 6];
        // Left:   row3 + row0
        // Right:  row3 - row0
        // Bottom: row3 + row1
        // Top:    row3 - row1
        // Near:   row3 + row2  (reverse-Z: near is at depth=1)
        // Far:    row3 - row2  (reverse-Z: far is at depth=0)
        for j in 0..4 {
            planes[0][j] = r3[j] + r0[j]; // left
            planes[1][j] = r3[j] - r0[j]; // right
            planes[2][j] = r3[j] + r1[j]; // bottom
            planes[3][j] = r3[j] - r1[j]; // top
            planes[4][j] = r3[j] + r2[j]; // near
            planes[5][j] = r3[j] - r2[j]; // far
        }
        // Normalize each plane.
        for plane in &mut planes {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if len > 1e-8 {
                for v in plane.iter_mut() { *v /= len; }
            }
        }
        FrustumPlanes { planes }
    }

    /// Test if a bounding sphere (center, radius) is at least partially inside the frustum.
    pub fn contains_sphere(&self, center: glam::Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane[0] * center.x + plane[1] * center.y + plane[2] * center.z + plane[3];
            if dist < -radius {
                return false; // fully outside this plane
            }
        }
        true
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectUniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],       // world-space model matrix for position/normal transform
    pub color: [f32; 4],            // rgb = base color, a = emissive flag (>0.5 = emissive)
    pub material: [f32; 4],         // x = metallic, y = roughness, zw = unused
    pub _pad0: [f32; 4], pub _pad1: [f32; 4], pub _pad2: [f32; 4], pub _pad3: [f32; 4], pub _pad4: [f32; 4], pub _pad5: [f32; 4], // 256 bytes total
}
const _: () = assert!(std::mem::size_of::<ObjectUniforms>() == 256);

/// Scene-wide lighting and camera data passed to the shader via a separate uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneLighting {
    pub sun_direction: [f32; 4], // xyz = direction, w = unused
    pub sun_color: [f32; 4],    // rgb = color, a = intensity
    pub ambient: [f32; 4],      // x = ambient level, yzw = unused
    pub camera_pos: [f32; 4],   // xyz = camera world pos (near origin), w = unused
    pub light_vp: [[f32; 4]; 4], // sun view-projection matrix for shadow mapping
}
const _: () = assert!(std::mem::size_of::<SceneLighting>() == 128);

pub const SHADOW_MAP_SIZE: u32 = 2048;

pub struct GpuState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub pipeline: wgpu::RenderPipeline,
    pub sphere_inside_pipeline: wgpu::RenderPipeline,
    pub shadow_pipeline: wgpu::RenderPipeline,
    pub sphere_vertex_buf: wgpu::Buffer,
    pub sphere_index_buf: wgpu::Buffer,
    pub sphere_index_count: u32,
    pub box_vertex_buf: wgpu::Buffer,
    pub box_index_buf: wgpu::Buffer,
    pub box_index_count: u32,
    pub cockpit_vertex_buf: wgpu::Buffer,
    pub cockpit_index_buf: wgpu::Buffer,
    pub cockpit_index_count: u32,
    pub ext_box_vertex_buf: wgpu::Buffer,
    pub ext_box_index_buf: wgpu::Buffer,
    pub ext_box_index_count: u32,
    pub depth_view: wgpu::TextureView,
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub scene_lighting_buf: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub shadow_texture_view: wgpu::TextureView,
    pub shadow_bind_group: wgpu::BindGroup,

    // Bind group layouts — stored for creating additional pipelines (block renderer).
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub scene_bind_group_layout: wgpu::BindGroupLayout,
    pub shadow_bind_group_layout: wgpu::BindGroupLayout,

    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    // Star field rendering.
    pub star_pipeline: wgpu::RenderPipeline,
    pub star_point_pipeline: wgpu::RenderPipeline,
    pub star_quad_vertex_buf: wgpu::Buffer,
    pub star_quad_index_buf: wgpu::Buffer,
    pub star_instance_buf: wgpu::Buffer,
    pub star_scene_uniform_buf: wgpu::Buffer,
    pub star_bind_group: wgpu::BindGroup,
}

/// Initialize GPU state: adapter, device, surface, pipelines, buffers, egui.
pub fn init_gpu(window: Arc<Window>) -> GpuState {
    let size = window.inner_size();
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();

    let (adapter, device, queue) = pollster::block_on(async {
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface), ..Default::default()
        }).await.expect("no GPU adapter");
        info!(adapter = ?adapter.get_info().name, "GPU");
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
            .await.expect("no device");
        (adapter, device, queue)
    });

    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats[0];
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
        width: size.width.max(1), height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0], view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // Uniform buffer.
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"), size: 256 * MAX_OBJECTS as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(256),
            }, count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bind_group_layout, entries: &[wgpu::BindGroupEntry {
            binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform_buf, offset: 0, size: wgpu::BufferSize::new(256),
            }),
        }],
    });

    // Scene-wide lighting uniform (group 1).
    let scene_lighting_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene_lighting"),
        size: std::mem::size_of::<SceneLighting>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let scene_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<SceneLighting>() as u64),
            },
            count: None,
        }],
    });
    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_bind_group"),
        layout: &scene_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: scene_lighting_buf.as_entire_binding(),
        }],
    });

    // Shadow map: 2048x2048 depth texture with texture binding for sampling in main pass.
    let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_map"),
        size: wgpu::Extent3d { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_texture_view = shadow_texture.create_view(&Default::default());

    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    // Shadow bind group layout (group 2 in main pipeline): shadow texture + comparison sampler.
    let shadow_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    });

    let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_bind_group"),
        layout: &shadow_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"), source: wgpu::ShaderSource::Wgsl(include_str!("sphere.wgsl").into()),
    });

    // Main pipeline layout: group 0 (per-object), group 1 (scene lighting), group 2 (shadow map).
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout, &scene_bind_group_layout, &shadow_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Shadow pipeline layout: only group 0 (per-object uniforms). Depth-only, no fragment.
    let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let depth_format = wgpu::TextureFormat::Depth32Float;
    let depth_view = create_depth_texture(&device, config.width, config.height, depth_format);

    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline"), layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"), buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format, depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: Default::default(), bias: Default::default(),
        }),
        multisample: Default::default(), multiview: None, cache: None,
    });

    // Pipeline variant for inside-sphere rendering (no backface culling).
    // Used when camera is inside a celestial body (e.g., standing on planet surface).
    let sphere_inside_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sphere_inside"), layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // no culling -- camera is inside the sphere
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format, depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: Default::default(), bias: Default::default(),
        }),
        multisample: Default::default(), multiview: None, cache: None,
    });

    // Shadow pipeline: depth-only rendering from the sun's perspective.
    // Uses its own pipeline layout (only group 0) and no fragment shader.
    let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow_pipeline"),
        layout: Some(&shadow_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }],
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
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual, // standard depth (not reverse-Z)
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,     // offset to reduce shadow acne
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });

    // Sphere mesh.
    let sphere = IcoSphere::generate(4);
    let sphere_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere_vb"), contents: bytemuck::cast_slice(&sphere.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let sphere_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere_ib"), contents: bytemuck::cast_slice(&sphere.indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Box mesh — interior (inward normals, for first-person view inside ship).
    let (box_verts, box_idxs) = generate_box_mesh(4.0, 3.0, 8.0);
    let box_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("box_vb"), contents: bytemuck::cast_slice(&box_verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let box_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("box_ib"), contents: bytemuck::cast_slice(&box_idxs),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Cockpit window (glass front wall, separate from opaque box).
    let (win_verts, win_idxs) = generate_cockpit_window(4.0, 3.0, 8.0);
    let cockpit_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cockpit_vb"), contents: bytemuck::cast_slice(&win_verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let cockpit_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cockpit_ib"), contents: bytemuck::cast_slice(&win_idxs),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Box mesh — exterior (outward normals, for third-person view of ship on surface).
    let ext_box_idxs: Vec<u32> = box_idxs.chunks(3)
        .flat_map(|tri| [tri[0], tri[2], tri[1]])
        .collect();
    let ext_box_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ext_box_vb"), contents: bytemuck::cast_slice(&box_verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ext_box_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ext_box_ib"), contents: bytemuck::cast_slice(&ext_box_idxs),
        usage: wgpu::BufferUsages::INDEX,
    });

    // ---- Star field pipeline ----
    let star_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("star_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("stars.wgsl").into()),
    });

    let star_scene_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("star_scene_uniforms"),
        size: std::mem::size_of::<StarSceneUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let star_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("star_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<StarSceneUniforms>() as u64),
            },
            count: None,
        }],
    });

    let star_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("star_bind_group"),
        layout: &star_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: star_scene_uniform_buf.as_entire_binding(),
        }],
    });

    let star_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("star_pipeline_layout"),
        bind_group_layouts: &[&star_bind_group_layout],
        push_constant_ranges: &[],
    });

    let star_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("star_pipeline"),
        layout: Some(&star_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &star_shader,
            entry_point: Some("vs_main"),
            buffers: &[
                // Per-vertex: quad corner position.
                wgpu::VertexBufferLayout {
                    array_stride: 8, // [f32; 2]
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                // Per-instance: star position + color.
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<StarInstance>() as u64, // 32 bytes
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                },
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &star_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    // Additive blending: each star adds light. Dense galactic
                    // plane regions accumulate into a natural Milky Way band.
                    // Fragment alpha is scaled low to prevent total saturation.
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // billboards face camera
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false, // stars don't write depth
            depth_compare: wgpu::CompareFunction::Always, // always draw (behind everything)
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });

    // Star quad: 4 vertices for a unit billboard.
    let star_quad_verts: [[f32; 2]; 4] = [
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0],
    ];
    let star_quad_idxs: [u32; 6] = [0, 1, 2, 0, 2, 3];

    let star_quad_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("star_quad_vb"),
        contents: bytemuck::cast_slice(&star_quad_verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let star_quad_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("star_quad_ib"),
        contents: bytemuck::cast_slice(&star_quad_idxs),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Pre-allocated instance buffer for up to MAX_STAR_INSTANCES stars.
    let star_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("star_instances"),
        size: (std::mem::size_of::<StarInstance>() * MAX_STAR_INSTANCES) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Point-based star pipeline: 1 pixel per star, no billboard, no overdraw.
    // Uses the same star_instance_buf but as per-vertex data with PointList topology.
    let star_point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("star_point_pipeline"),
        layout: Some(&star_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &star_shader,
            entry_point: Some("vs_point"),
            buffers: &[
                // Per-vertex: star position + color (same layout as instance data).
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<StarInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                },
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &star_shader,
            entry_point: Some("fs_point"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });

    // egui.
    let egui_ctx = egui::Context::default();
    let egui_winit = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
    let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

    GpuState {
        surface, device, queue, config, pipeline, sphere_inside_pipeline, shadow_pipeline,
        sphere_vertex_buf, sphere_index_buf, sphere_index_count: sphere.indices.len() as u32,
        box_vertex_buf, box_index_buf, box_index_count: box_idxs.len() as u32,
        cockpit_vertex_buf, cockpit_index_buf, cockpit_index_count: win_idxs.len() as u32,
        ext_box_vertex_buf, ext_box_index_buf, ext_box_index_count: ext_box_idxs.len() as u32,
        depth_view, uniform_buf, bind_group,
        scene_lighting_buf, scene_bind_group,
        shadow_texture_view, shadow_bind_group,
        bind_group_layout, scene_bind_group_layout, shadow_bind_group_layout,
        egui_ctx, egui_winit, egui_renderer,
        star_pipeline, star_point_pipeline, star_quad_vertex_buf, star_quad_index_buf,
        star_instance_buf, star_scene_uniform_buf, star_bind_group,
    }
}

pub fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    texture.create_view(&Default::default())
}

/// Generate box mesh (interior visible -- normals face inward).
pub fn generate_box_mesh(width: f32, height: f32, length: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let hw = width / 2.0;
    let hl = length / 2.0;

    // Door hole in the right wall (x=+hw), centered at z=0.
    // Door dimensions: 1.2m wide (z: -0.6 to 0.6), 2.1m tall (y: 0 to 2.1).
    let dz = 0.6_f32;  // half-width of door
    let dh = 2.1_f32;  // door height

    let vertices: Vec<[f32; 3]> = vec![
        // 0-3: Floor (y=0)
        [-hw, 0.0, -hl], [hw, 0.0, -hl], [hw, 0.0, hl], [-hw, 0.0, hl],
        // 4-7: Ceiling (y=height)
        [-hw, height, -hl], [hw, height, -hl], [hw, height, hl], [-hw, height, hl],
        // 8-11: Left wall (x=-hw)
        [-hw, 0.0, -hl], [-hw, 0.0, hl], [-hw, height, hl], [-hw, height, -hl],
        // 12-19: Right wall with door hole -- 4 quads around the opening.
        //   Below door (y: 0 to 0 -- skip, door goes to floor)
        //   Left of door (z: -hl to -dz)
        [hw, 0.0, -hl], [hw, 0.0, -dz], [hw, height, -dz], [hw, height, -hl],   // 12-15
        //   Right of door (z: +dz to +hl)
        [hw, 0.0, dz], [hw, 0.0, hl], [hw, height, hl], [hw, height, dz],        // 16-19
        //   Above door (z: -dz to +dz, y: dh to height)
        [hw, dh, -dz], [hw, dh, dz], [hw, height, dz], [hw, height, -dz],        // 20-23
        // 24-27: Back wall (z=+hl)
        [-hw, 0.0, hl], [hw, 0.0, hl], [hw, height, hl], [-hw, height, hl],
        // Front wall (cockpit window) is a SEPARATE mesh with glass material.
    ];

    // Inward-facing triangles (CW from outside = CCW from inside).
    let indices: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2,       // floor
        4, 5, 6, 4, 6, 7,       // ceiling
        8, 10, 9, 8, 11, 10,    // left wall
        12, 13, 14, 12, 14, 15, // right wall: left of door
        16, 17, 18, 16, 18, 19, // right wall: right of door
        20, 21, 22, 20, 22, 23, // right wall: above door
        24, 25, 26, 24, 26, 27, // back wall
    ];
    (vertices, indices)
}

/// Generate cockpit window quad (front wall, z=-length/2, inward-facing).
/// Rendered separately from the box mesh with a glass material.
pub fn generate_cockpit_window(width: f32, height: f32, length: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let hw = width / 2.0;
    let hl = length / 2.0;
    let vertices = vec![
        [-hw, 0.0, -hl], [hw, 0.0, -hl], [hw, height, -hl], [-hw, height, -hl],
    ];
    // Inward-facing (same winding as interior box).
    let indices = vec![0, 2, 1, 0, 3, 2];
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn test_frustum() -> FrustumPlanes {
        // Camera at origin, looking down -Z, 90° FOV, 1:1 aspect.
        let proj = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 1000.0);
        let view = Mat4::look_to_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        FrustumPlanes::from_vp(&(proj * view))
    }

    #[test]
    fn sphere_in_front_is_visible() {
        let frustum = test_frustum();
        // Sphere at z=-10 (in front of camera), radius 1.
        assert!(frustum.contains_sphere(Vec3::new(0.0, 0.0, -10.0), 1.0));
    }

    #[test]
    fn sphere_behind_camera_is_culled() {
        let frustum = test_frustum();
        // Sphere at z=+10 (behind camera), radius 1.
        assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, 10.0), 1.0));
    }

    #[test]
    fn sphere_far_left_is_culled() {
        let frustum = test_frustum();
        // Sphere far to the left at z=-10, well outside 90° FOV.
        assert!(!frustum.contains_sphere(Vec3::new(-50.0, 0.0, -10.0), 1.0));
    }

    #[test]
    fn sphere_far_right_is_culled() {
        let frustum = test_frustum();
        assert!(!frustum.contains_sphere(Vec3::new(50.0, 0.0, -10.0), 1.0));
    }

    #[test]
    fn sphere_above_is_culled() {
        let frustum = test_frustum();
        assert!(!frustum.contains_sphere(Vec3::new(0.0, 50.0, -10.0), 1.0));
    }

    #[test]
    fn large_sphere_partially_in_view_is_visible() {
        let frustum = test_frustum();
        // Sphere center is outside left, but radius is large enough to overlap.
        assert!(frustum.contains_sphere(Vec3::new(-12.0, 0.0, -10.0), 5.0));
    }

    #[test]
    fn sphere_at_origin_is_visible() {
        let frustum = test_frustum();
        // Sphere at camera position — should be visible (camera is inside it).
        assert!(frustum.contains_sphere(Vec3::ZERO, 5.0));
    }

    #[test]
    fn sphere_beyond_far_plane_is_culled() {
        let frustum = test_frustum();
        // Sphere way past far plane (1000.0).
        assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, -2000.0), 1.0));
    }

    #[test]
    fn sphere_on_edge_of_view_is_visible() {
        let frustum = test_frustum();
        // 90° FOV means at z=-10, the visible width is ~10 each side.
        // Sphere at x=9 with radius 2 should still be partially visible.
        assert!(frustum.contains_sphere(Vec3::new(9.0, 0.0, -10.0), 2.0));
    }
}
