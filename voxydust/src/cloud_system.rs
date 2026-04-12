//! Cloud system — manages noise textures, cloud uniforms, and compute dispatch
//! for Nubis-style volumetric cloud rendering.
//!
//! Noise textures are generated once per planet load via compute shader.
//! Cloud parameters are uploaded each frame as a uniform buffer.

use wgpu;

use voxeldust_core::system::PlanetParams;

/// Cloud noise texture sizes.
pub const CLOUD_SHAPE_SIZE: u32 = 128;
pub const CLOUD_DETAIL_SIZE: u32 = 32;

/// GPU uniform for cloud rendering. Uploaded each frame when near a cloudy planet.
/// All values derived from CloudParams (seed-based) + game state.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudUniforms {
    /// Observer position relative to planet center (km). w = game_time (seconds).
    pub observer_pos: [f32; 4],
    /// x = planet radius (km), y = cloud base (km), z = cloud thickness (km), w = enabled.
    pub geometry: [f32; 4],
    /// x = base coverage, y = density_scale, z = cloud_type, w = absorption_factor.
    pub density_params: [f32; 4],
    /// xyz = wind velocity (km/s), w = wind_shear.
    pub wind: [f32; 4],
    /// xyz = scatter color (linear RGB), w = weather_scale (km).
    pub scatter: [f32; 4],
    /// xyz = sun direction (toward sun), w = sun intensity.
    pub sun: [f32; 4],
    /// xyz = sun color (linear RGB), w = noise base frequency.
    pub sun_color: [f32; 4],
    /// x = shape noise scale (1/km), y = detail noise scale (1/km), z = weather_octaves, w = unused.
    pub noise_params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<CloudUniforms>() == 128);

/// Noise generation parameters uniform.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NoiseGenParams {
    pub seed: [u32; 4],       // x = planet seed lo, y = planet seed hi, z = texture_size, w = pass_type
    pub frequency: [f32; 4],  // x = base_frequency, y = persistence, z = lacunarity, w = unused
}

/// Manages cloud noise textures, weather map, and compute pipelines.
pub struct CloudSystem {
    // Noise textures (generated per planet load).
    pub shape_texture: Option<wgpu::Texture>,
    pub shape_view: Option<wgpu::TextureView>,
    pub detail_texture: Option<wgpu::Texture>,
    pub detail_view: Option<wgpu::TextureView>,

    pub noise_sampler: wgpu::Sampler,

    // Compute pipelines for noise generation.
    shape_pipeline: wgpu::ComputePipeline,
    detail_pipeline: wgpu::ComputePipeline,
    noise_bind_group_layout: wgpu::BindGroupLayout,
    noise_params_buf: wgpu::Buffer,

    // Cloud uniform buffer (updated per frame).
    pub cloud_uniform_buf: wgpu::Buffer,

    /// The planet seed for which noise was last generated. Used to detect planet switch.
    current_planet_seed: Option<u64>,

    // Weather map (2D texture, updated asynchronously per game-minute).
    pub weather_map_texture: Option<wgpu::Texture>,
    pub weather_map_view: Option<wgpu::TextureView>,
    /// Background thread handle for weather map generation.
    weather_gen_handle: Option<std::thread::JoinHandle<voxeldust_core::weather::WeatherMap>>,
    /// Game time at which the current weather map was generated.
    last_weather_time: f64,
    /// Whether the weather map has been uploaded to GPU at least once.
    pub weather_map_ready: bool,
    /// Fallback 1×1 dummy 2D texture view for when weather map isn't ready.
    pub dummy_2d_view: wgpu::TextureView,
}

impl CloudSystem {
    pub fn new(device: &wgpu::Device) -> Self {
        let noise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud_noise_gen"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cloud_noise_gen.wgsl").into()),
        });

        let noise_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_noise_layout"),
            entries: &[
                // binding 0: output 3D storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // binding 1: noise params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<NoiseGenParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let noise_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud_noise_pipeline_layout"),
            bind_group_layouts: &[&noise_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shape_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud_shape_noise"),
            layout: Some(&noise_pipeline_layout),
            module: &noise_shader,
            entry_point: Some("gen_shape"),
            compilation_options: Default::default(),
            cache: None,
        });

        let detail_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud_detail_noise"),
            layout: Some(&noise_pipeline_layout),
            module: &noise_shader,
            entry_point: Some("gen_detail"),
            compilation_options: Default::default(),
            cache: None,
        });

        let noise_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud_noise_params"),
            size: std::mem::size_of::<NoiseGenParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud_noise_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let cloud_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud_uniforms"),
            size: std::mem::size_of::<CloudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shape_texture: None,
            shape_view: None,
            detail_texture: None,
            detail_view: None,
            noise_sampler,
            shape_pipeline,
            detail_pipeline,
            noise_bind_group_layout,
            noise_params_buf,
            cloud_uniform_buf,
            current_planet_seed: None,
            weather_map_texture: None,
            weather_map_view: None,
            weather_gen_handle: None,
            last_weather_time: -1000.0, // force immediate generation
            weather_map_ready: false,
            dummy_2d_view: {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("dummy_weather_2d"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                tex.create_view(&Default::default())
            },
        }
    }

    /// Get the weather map view, falling back to a 1×1 dummy if not ready.
    pub fn weather_map_view_or_dummy(&self, _device: &wgpu::Device) -> &wgpu::TextureView {
        self.weather_map_view.as_ref().unwrap_or(&self.dummy_2d_view)
    }

    /// Kick off asynchronous weather map generation for a planet.
    /// Non-blocking: spawns a background thread. Call `poll_weather_map()` each
    /// frame to check for completion and upload to GPU.
    ///
    /// Weather map is generated ONCE per planet (static large-scale pattern).
    /// Cloud animation comes from wind scrolling the 3D noise in the shader.
    /// Only regenerates when switching to a different planet.
    pub fn request_weather_update(
        &mut self,
        planet: &PlanetParams,
        _game_time: f64,
    ) {
        // Don't start if already generating.
        if self.weather_gen_handle.is_some() { return; }

        // Only generate once per planet — skip if already generated for this seed.
        if self.weather_map_ready && self.current_planet_seed == Some(planet.planet_seed) { return; }

        let planet_clone = planet.clone();
        // Use time=0 for static weather pattern. Cloud motion comes from wind scrolling.
        self.weather_gen_handle = Some(std::thread::spawn(move || {
            voxeldust_core::weather::WeatherMap::generate(&planet_clone, 0.0)
        }));
    }

    /// Check if background weather generation is complete. If so, upload to GPU.
    /// Returns true if a new weather map was uploaded this frame.
    pub fn poll_weather_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> bool {
        let handle = match self.weather_gen_handle.take() {
            Some(h) if h.is_finished() => h,
            Some(h) => { self.weather_gen_handle = Some(h); return false; }
            None => return false,
        };

        let weather_map = match handle.join() {
            Ok(map) => map,
            Err(_) => { tracing::error!("weather generation thread panicked"); return false; }
        };

        // Create or reuse the weather map texture.
        let needs_create = self.weather_map_texture.is_none();
        if needs_create {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("weather_map"),
                size: wgpu::Extent3d {
                    width: weather_map.width,
                    height: weather_map.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.weather_map_texture = Some(tex);
            self.weather_map_view = Some(view);
        }

        // Upload weather data to GPU.
        if let Some(tex) = &self.weather_map_texture {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &weather_map.data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(weather_map.width * 4),
                    rows_per_image: Some(weather_map.height),
                },
                wgpu::Extent3d {
                    width: weather_map.width,
                    height: weather_map.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.weather_map_ready = true;
        tracing::info!(
            width = weather_map.width,
            height = weather_map.height,
            "uploaded weather map to GPU"
        );
        needs_create // true if bind group needs rebuild (new texture)
    }

    /// Generate noise textures for a planet if not already generated for this seed.
    /// Dispatches compute shaders. Call once when planet changes.
    pub fn ensure_noise_for_planet(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet_seed: u64,
    ) {
        if self.current_planet_seed == Some(planet_seed) {
            return; // Already generated for this planet.
        }
        tracing::info!(planet_seed, "starting cloud noise generation");
        self.current_planet_seed = Some(planet_seed);

        // Create shape texture (128³).
        let shape_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_shape"),
            size: wgpu::Extent3d {
                width: CLOUD_SHAPE_SIZE,
                height: CLOUD_SHAPE_SIZE,
                depth_or_array_layers: CLOUD_SHAPE_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shape_view = shape_tex.create_view(&Default::default());

        // Create detail texture (32³).
        let detail_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_detail"),
            size: wgpu::Extent3d {
                width: CLOUD_DETAIL_SIZE,
                height: CLOUD_DETAIL_SIZE,
                depth_or_array_layers: CLOUD_DETAIL_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let detail_view = detail_tex.create_view(&Default::default());

        // Dispatch shape noise generation.
        let seed_lo = (planet_seed & 0xFFFFFFFF) as u32;
        let seed_hi = (planet_seed >> 32) as u32;

        // Shape pass.
        let shape_params = NoiseGenParams {
            seed: [seed_lo, seed_hi, CLOUD_SHAPE_SIZE, 0],
            frequency: [8.0, 0.5, 2.0, 0.0], // base_freq, persistence, lacunarity
        };
        queue.write_buffer(&self.noise_params_buf, 0, bytemuck::bytes_of(&shape_params));

        let shape_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud_shape_bg"),
            layout: &self.noise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shape_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.noise_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloud_shape_gen"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud_shape"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shape_pipeline);
            pass.set_bind_group(0, &shape_bg, &[]);
            let wg = (CLOUD_SHAPE_SIZE + 3) / 4;
            pass.dispatch_workgroups(wg, wg, wg);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Detail pass.
        let detail_params = NoiseGenParams {
            seed: [seed_lo.wrapping_add(12345), seed_hi, CLOUD_DETAIL_SIZE, 1],
            frequency: [4.0, 0.5, 2.0, 0.0],
        };
        queue.write_buffer(&self.noise_params_buf, 0, bytemuck::bytes_of(&detail_params));

        let detail_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud_detail_bg"),
            layout: &self.noise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&detail_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.noise_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloud_detail_gen"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud_detail"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.detail_pipeline);
            pass.set_bind_group(0, &detail_bg, &[]);
            let wg = (CLOUD_DETAIL_SIZE + 3) / 4;
            pass.dispatch_workgroups(wg, wg, wg);
        }
        queue.submit(std::iter::once(encoder.finish()));

        tracing::info!(planet_seed, "generated cloud noise textures");

        self.shape_texture = Some(shape_tex);
        self.shape_view = Some(shape_view);
        self.detail_texture = Some(detail_tex);
        self.detail_view = Some(detail_view);
    }

    /// Build CloudUniforms from planet parameters and current state.
    pub fn build_uniforms(
        planet: &PlanetParams,
        observer_km: [f32; 3],
        game_time: f64,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        sun_color: [f32; 3],
    ) -> CloudUniforms {
        let clouds = &planet.clouds;
        let planet_r_km = (planet.radius_m / 1000.0) as f32;
        let cloud_base_km = (clouds.cloud_base_altitude / 1000.0) as f32;
        let cloud_thickness_km = (clouds.cloud_layer_thickness / 1000.0) as f32;

        // Wind in km/s (convert from m/s).
        let wind_km = [
            (clouds.wind_velocity[0] / 1000.0) as f32,
            (clouds.wind_velocity[1] / 1000.0) as f32,
            (clouds.wind_velocity[2] / 1000.0) as f32,
        ];

        // Noise scale: controls individual cloud size.
        // Individual clouds should be ~5-20km across. The 128³ noise texture tiles
        // at this scale, so one tile = one "cloud field" area.
        // shape_scale = 1/cloud_field_size_km: how many tiles per km.
        let cloud_field_km = 30.0_f32; // one noise tile covers 30km — produces ~5-15km clouds
        let shape_scale = 1.0 / cloud_field_km;
        let detail_scale = shape_scale * 4.0; // detail 4× finer than shape

        CloudUniforms {
            observer_pos: [observer_km[0], observer_km[1], observer_km[2], game_time as f32],
            geometry: [planet_r_km, cloud_base_km, cloud_thickness_km, 1.0],
            density_params: [
                clouds.base_coverage as f32,
                clouds.density_scale as f32,
                clouds.cloud_type as f32,
                clouds.absorption_factor as f32,
            ],
            wind: [wind_km[0], wind_km[1], wind_km[2], clouds.wind_shear as f32],
            scatter: [
                clouds.scatter_color[0],
                clouds.scatter_color[1],
                clouds.scatter_color[2],
                (clouds.weather_scale / 1000.0) as f32, // km
            ],
            sun: [sun_dir[0], sun_dir[1], sun_dir[2], sun_intensity],
            sun_color: [sun_color[0], sun_color[1], sun_color[2], 8.0], // w = base noise frequency
            noise_params: [shape_scale, detail_scale, clouds.weather_octaves as f32, 0.0],
        }
    }

    /// Whether noise textures are ready for rendering.
    pub fn has_noise(&self) -> bool {
        self.shape_view.is_some() && self.detail_view.is_some()
    }
}
