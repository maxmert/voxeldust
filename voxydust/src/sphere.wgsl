// PBR sphere rendering shader with Cook-Torrance BRDF and voxel ray-marched shadows.

const PI: f32 = 3.14159265358979;

struct Uniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    color: vec4<f32>,       // rgb = base color, a = emissive flag
    material: vec4<f32>,    // x = metallic, y = roughness
}

struct SceneLighting {
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,    // rgb = color, a = intensity
    ambient: vec4<f32>,      // x = ambient level
    camera_pos: vec4<f32>,   // xyz = camera world pos
}

struct RenderConfig {
    shadow_max_steps: u32,
    volume_size: f32,
    inv_volume_size: f32,
    block_light_enabled: u32,
    atmosphere_samples: u32,
    god_ray_samples: u32,
    voxel_shadows_enabled: u32,
    eclipse_shadows_enabled: u32,
    frame_count: u32,
    soft_shadows_enabled: u32,
    _pad0: u32,
    _pad1: u32,
}

struct VoxelVolumeParams {
    world_to_volume: mat4x4<f32>,   // transforms world_pos to volume-index coords (0..size)
    sun_dir_and_size: vec4<f32>,    // xyz = sun dir in volume space, w = volume size
    inv_volume_size: vec4<f32>,     // x = 1/size
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

@group(1) @binding(0)
var<uniform> scene: SceneLighting;

@group(1) @binding(1)
var<uniform> config: RenderConfig;

@group(2) @binding(0)
var voxel_occupancy: texture_3d<u32>;

@group(2) @binding(1)
var voxel_light_tex: texture_3d<f32>;

@group(2) @binding(2)
var light_sampler: sampler;

@group(2) @binding(3)
var<uniform> voxel_params: VoxelVolumeParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4(in.position, 1.0);
    // Transform normal and position to world space.
    out.world_normal = normalize((u.model * vec4(in.position, 0.0)).xyz);
    out.world_pos = (u.model * vec4(in.position, 1.0)).xyz;
    return out;
}

// ---------------------------------------------------------------------------
// PBR functions
// ---------------------------------------------------------------------------

// GGX/Trowbridge-Reitz normal distribution.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

// Schlick-GGX geometry function.
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith's method combining both view and light geometry.
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

// Schlick Fresnel approximation.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// ---------------------------------------------------------------------------
// Voxel sun shadow: Amanatides & Woo DDA through 3D occupancy volume
// ---------------------------------------------------------------------------

fn voxel_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, clip_pos: vec4<f32>) -> f32 {
    if config.voxel_shadows_enabled == 0u { return 1.0; }

    // Normal bias: offset ray start along surface normal to avoid self-shadowing.
    let biased_pos = world_pos + world_normal * 0.5;

    let vol_size = voxel_params.sun_dir_and_size.w;

    // Transform biased position to volume-index coordinates.
    let vol_pos_base = (voxel_params.world_to_volume * vec4(biased_pos, 1.0)).xyz;

    // Temporal jitter for soft shadow edges.
    var vol_pos = vol_pos_base;
    if config.soft_shadows_enabled != 0u {
        let jitter_idx = (u32(clip_pos.x) + u32(clip_pos.y) * 2u + config.frame_count) % 4u;
        let jitter = array<vec3<f32>, 4>(
            vec3(0.15, 0.15, 0.0),
            vec3(-0.15, 0.15, 0.0),
            vec3(0.15, -0.15, 0.0),
            vec3(-0.15, -0.15, 0.0)
        );
        vol_pos = vol_pos_base + jitter[jitter_idx];
    }

    // Outside volume: no shadow data — return lit.
    if any(vol_pos < vec3(0.0)) || any(vol_pos >= vec3(vol_size)) { return 1.0; }

    // DDA ray march in volume space using volume-space sun direction.
    let vol_sun_dir = voxel_params.sun_dir_and_size.xyz;
    let step_dir = sign(vol_sun_dir);
    var voxel = vec3<i32>(floor(vol_pos));
    let inv_dir = 1.0 / max(abs(vol_sun_dir), vec3(1e-10));
    var t_max = ((vec3<f32>(voxel) + max(step_dir, vec3(0.0))) - vol_pos) * inv_dir;
    let t_delta = abs(inv_dir);
    let vol_size_i = i32(vol_size);
    let max_steps = i32(config.shadow_max_steps);

    for (var i = 0; i < max_steps; i++) {
        if t_max.x < t_max.y && t_max.x < t_max.z {
            voxel.x += i32(step_dir.x);
            t_max.x += t_delta.x;
        } else if t_max.y < t_max.z {
            voxel.y += i32(step_dir.y);
            t_max.y += t_delta.y;
        } else {
            voxel.z += i32(step_dir.z);
            t_max.z += t_delta.z;
        }
        if any(voxel < vec3(0)) || any(voxel >= vec3(vol_size_i)) { break; }
        let occ = textureLoad(voxel_occupancy, voxel, 0).r;
        if occ > 128u { return 0.0; }
    }
    return 1.0;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = u.color.rgb;

    // Emissive objects (stars, markers with alpha > 0.5).
    if u.color.a > 0.5 {
        return vec4(base_color, 1.0);
    }

    let metallic = u.material.x;
    let roughness = max(u.material.y, 0.04); // clamp to avoid divide-by-zero
    let glass = u.material.z;

    // Glass material: screen-door transparency (discard checkerboard pattern).
    if glass > 0.5 {
        let screen_pos = vec2<i32>(in.clip_position.xy);
        if (screen_pos.x + screen_pos.y) % 2 == 0 {
            discard;
        }
    }

    let N = normalize(in.world_normal);
    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let L = normalize(scene.sun_direction.xyz);
    let H = normalize(V + L);

    let n_dot_l = max(dot(N, L), 0.0);
    let n_dot_v = max(dot(N, V), 0.001);
    let n_dot_h = max(dot(N, H), 0.0);
    let v_dot_h = max(dot(V, H), 0.0);

    // Fresnel: dielectric F0 = 0.04, metallic F0 = base_color.
    let f0 = mix(vec3(0.04), base_color, metallic);
    let F = fresnel_schlick(v_dot_h, f0);

    // Distribution and geometry.
    let D = distribution_ggx(n_dot_h, roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, roughness);

    // Cook-Torrance specular.
    let specular = (D * F * G) / (4.0 * n_dot_v * n_dot_l + 0.0001);

    // Diffuse (energy-conserving).
    let k_d = (1.0 - F) * (1.0 - metallic);
    let diffuse = k_d * base_color / PI;

    let sun_intensity = scene.sun_color.a;
    let light_color = scene.sun_color.rgb * sun_intensity;

    // Voxel ray-marched sun shadow (replaces shadow map).
    let shadow = voxel_shadow(in.world_pos, N, in.clip_position);

    // Block light contribution (from emissive voxels).
    var block_light = vec3(0.0);
    if config.block_light_enabled != 0u {
        let vol_size = voxel_params.sun_dir_and_size.w;
        let vol_pos = (voxel_params.world_to_volume * vec4(in.world_pos, 1.0)).xyz;
        let vol_uv = vol_pos / vol_size;
        if all(vol_uv >= vec3(0.0)) && all(vol_uv < vec3(1.0)) {
            block_light = textureSample(voxel_light_tex, light_sampler, vol_uv).rgb;
        }
    }

    let sun_contrib = (diffuse + specular) * light_color * n_dot_l * shadow;
    let block_contrib = base_color * block_light;

    // Ambient.
    let ambient_color = scene.ambient.x * base_color * 0.3;

    // ACES filmic tonemapping.
    var color = ambient_color + sun_contrib + block_contrib;
    color = color * (color * 2.51 + vec3(0.03)) / (color * (color * 2.43 + vec3(0.59)) + vec3(0.14));

    return vec4(color, 1.0);
}
