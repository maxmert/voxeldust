//! Graphics quality settings — configurable per-feature toggles and quality knobs.
//!
//! Stored as a bevy_ecs Resource, read by the render pipeline each frame.
//! Adjustable at runtime via game variables now, UI menu later.

use bevy_ecs::prelude::*;

/// All lighting/rendering quality knobs. Adjustable at runtime.
#[derive(Resource, Clone, Debug)]
pub struct GraphicsSettings {
    // -- Voxel shadows --
    /// Master toggle for voxel ray-marched shadows. When false, surfaces use
    /// simple N·L lighting with no shadows (zero GPU cost).
    pub voxel_shadows_enabled: bool,
    /// Resolution of the 3D voxel volume (one axis). Must be 64, 128, or 256.
    /// Each voxel = 1 block = 1m. Volume covers N metres in each direction from
    /// the player, so 128 → 128m shadow range.
    pub voxel_volume_size: u32,
    /// Maximum DDA ray march steps per pixel for sun shadows.
    /// Higher = longer shadow range within the volume, more expensive.
    pub shadow_max_steps: u32,
    /// Render shadow factor at half resolution and bilateral-upscale.
    /// Roughly halves shadow GPU cost.
    pub shadow_half_res: bool,
    /// Temporal jitter for soft shadow penumbra (accumulates over 4 frames).
    pub soft_shadows: bool,

    // -- Block light --
    /// Enable emissive block light propagation (lava, energy crystals, etc.).
    pub block_light_enabled: bool,
    /// Number of flood-fill iterations for block light propagation.
    /// Higher = light reaches further from emissive blocks (max 15 = full range).
    pub block_light_iterations: u32,

    // -- Atmosphere --
    /// Enable atmospheric scattering (Rayleigh/Mie) on planets with atmospheres.
    pub atmosphere_enabled: bool,
    /// Number of primary ray march samples for atmosphere scattering.
    pub atmosphere_samples: u32,
    /// Render atmosphere at half resolution and bilateral-upscale.
    pub atmosphere_half_res: bool,

    // -- God rays --
    /// Enable screen-space volumetric light shafts from the star.
    pub god_rays_enabled: bool,
    /// Number of radial blur samples for god rays.
    pub god_ray_samples: u32,

    // -- Eclipse shadows --
    /// Enable analytical eclipse shadows for distant celestial body occultation.
    pub eclipse_shadows_enabled: bool,

    // -- Volumetric clouds --
    /// Enable volumetric cloud rendering (Nubis technique).
    pub cloud_enabled: bool,
    /// Cloud ray march step count. Higher = better quality, more expensive.
    /// Low=32, Medium=48, High=64, Ultra=96.
    pub cloud_steps: u32,
    /// Enable cloud shadow map (Beer Shadow Map) for terrain darkening under clouds.
    pub cloud_shadows_enabled: bool,

    // -- HDR + Tonemapping --
    /// Enable HDR render target with separate tonemapping pass.
    /// When false, ACES tonemapping is applied directly in fragment shaders.
    pub hdr_enabled: bool,
}

impl GraphicsSettings {
    /// Minimal quality — for low-end hardware or maximum FPS.
    pub fn low() -> Self {
        Self {
            voxel_shadows_enabled: true,
            voxel_volume_size: 64,
            shadow_max_steps: 32,
            shadow_half_res: true,
            soft_shadows: false,

            block_light_enabled: true,
            block_light_iterations: 8,

            atmosphere_enabled: false,
            atmosphere_samples: 8,
            atmosphere_half_res: true,

            god_rays_enabled: false,
            god_ray_samples: 32,

            eclipse_shadows_enabled: false,

            cloud_enabled: false,
            cloud_steps: 32,
            cloud_shadows_enabled: false,

            hdr_enabled: false,
        }
    }

    /// Balanced quality — good visuals with solid performance.
    pub fn medium() -> Self {
        Self {
            voxel_shadows_enabled: true,
            voxel_volume_size: 128,
            shadow_max_steps: 64,
            shadow_half_res: true,
            soft_shadows: true,

            block_light_enabled: true,
            block_light_iterations: 12,

            atmosphere_enabled: true,
            atmosphere_samples: 8,
            atmosphere_half_res: true,

            god_rays_enabled: false,
            god_ray_samples: 48,

            eclipse_shadows_enabled: true,

            cloud_enabled: true,
            cloud_steps: 48,
            cloud_shadows_enabled: false,

            hdr_enabled: true,
        }
    }

    /// High quality — all features enabled, full resolution.
    pub fn high() -> Self {
        Self {
            voxel_shadows_enabled: true,
            voxel_volume_size: 128,
            shadow_max_steps: 96,
            shadow_half_res: false,
            soft_shadows: true,

            block_light_enabled: true,
            block_light_iterations: 15,

            atmosphere_enabled: true,
            atmosphere_samples: 16,
            atmosphere_half_res: false,

            god_rays_enabled: true,
            god_ray_samples: 64,

            eclipse_shadows_enabled: true,

            cloud_enabled: true,
            cloud_steps: 64,
            cloud_shadows_enabled: true,

            hdr_enabled: true,
        }
    }

    /// Maximum quality — larger volume, maximum samples.
    pub fn ultra() -> Self {
        Self {
            voxel_shadows_enabled: true,
            voxel_volume_size: 256,
            shadow_max_steps: 96,
            shadow_half_res: false,
            soft_shadows: true,

            block_light_enabled: true,
            block_light_iterations: 15,

            atmosphere_enabled: true,
            atmosphere_samples: 32,
            atmosphere_half_res: false,

            god_rays_enabled: true,
            god_ray_samples: 96,

            eclipse_shadows_enabled: true,

            cloud_enabled: true,
            cloud_steps: 96,
            cloud_shadows_enabled: true,

            hdr_enabled: true,
        }
    }

    /// Clamp all values to valid ranges.
    pub fn validate(&mut self) {
        self.voxel_volume_size = match self.voxel_volume_size {
            0..=96 => 64,
            97..=192 => 128,
            _ => 256,
        };
        self.shadow_max_steps = self.shadow_max_steps.clamp(16, 128);
        self.block_light_iterations = self.block_light_iterations.clamp(1, 15);
        self.atmosphere_samples = self.atmosphere_samples.clamp(4, 64);
        self.god_ray_samples = self.god_ray_samples.clamp(16, 128);
        self.cloud_steps = self.cloud_steps.clamp(16, 128);
    }
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self::medium()
    }
}

/// GPU-side render configuration uniform. Mirrors the subset of [`GraphicsSettings`]
/// that shaders need to read each frame. Packed to 32 bytes (2 × vec4<f32>).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderConfig {
    /// Maximum DDA steps for voxel sun shadow ray march.
    pub shadow_max_steps: u32,
    /// Volume size in voxels (one axis). 64, 128, or 256.
    pub volume_size: f32,
    /// 1.0 / volume_size.
    pub inv_volume_size: f32,
    /// 1 = block light enabled, 0 = disabled.
    pub block_light_enabled: u32,

    /// Number of atmosphere ray march samples.
    pub atmosphere_samples: u32,
    /// Number of god ray radial blur samples.
    pub god_ray_samples: u32,
    /// 1 = voxel shadows enabled, 0 = disabled.
    pub voxel_shadows_enabled: u32,
    /// 1 = eclipse shadows enabled, 0 = disabled.
    pub eclipse_shadows_enabled: u32,

    /// Frame counter for temporal effects (shadow jitter, TAA).
    pub frame_count: u32,
    /// 1 = soft shadows enabled (temporal jitter), 0 = hard shadows.
    pub soft_shadows_enabled: u32,
    /// 1 = HDR pipeline enabled (render to Rgba16Float, tonemap in composite pass).
    pub hdr_enabled: u32,
    /// Padding to 48 bytes (3 × vec4).
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<RenderConfig>() == 48);

impl RenderConfig {
    /// Build from the current graphics settings and frame number.
    pub fn from_settings(s: &GraphicsSettings, frame_count: u64) -> Self {
        Self {
            shadow_max_steps: s.shadow_max_steps,
            volume_size: s.voxel_volume_size as f32,
            inv_volume_size: 1.0 / s.voxel_volume_size as f32,
            block_light_enabled: s.block_light_enabled as u32,
            atmosphere_samples: s.atmosphere_samples,
            god_ray_samples: s.god_ray_samples,
            voxel_shadows_enabled: s.voxel_shadows_enabled as u32,
            eclipse_shadows_enabled: s.eclipse_shadows_enabled as u32,
            frame_count: (frame_count % 16) as u32,
            soft_shadows_enabled: s.soft_shadows as u32,
            hdr_enabled: s.hdr_enabled as u32,
            _pad: 0,
        }
    }
}
