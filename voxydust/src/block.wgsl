// Block mesh rendering shader — PBR Cook-Torrance BRDF with voxel ray-marched shadows.
//
// Differs from sphere.wgsl in vertex inputs: blocks have explicit per-vertex
// normals and colors (vs spheres which derive normals from position).
// The fragment shader shares the same PBR lighting model.

const PI: f32 = 3.14159265358979;

struct Uniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    color: vec4<f32>,       // rgb = tint (multiplied with vertex color), a = emissive flag
    material: vec4<f32>,    // x = metallic, y = roughness, z = glass flag
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
    hdr_enabled: u32,
    _pad0: u32,
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

@group(2) @binding(4)
var shadow_map: texture_depth_2d_array;

@group(2) @binding(5)
var shadow_sampler: sampler_comparison;

struct ShadowCascades {
    light_vp: array<mat4x4<f32>, 4>,
    splits: vec4<f32>,
    _pad: vec4<f32>,
}

@group(2) @binding(6)
var<uniform> cascades: ShadowCascades;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(3) view_z: f32,
    @location(2) vertex_color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4(in.position, 1.0);
    // Transform normal to world space (using model matrix upper-3x3).
    out.world_normal = normalize((u.model * vec4(in.normal, 0.0)).xyz);
    out.world_pos = (u.model * vec4(in.position, 1.0)).xyz;
    // View-space Z for cascade selection (distance from camera along view forward).
    // In floating-origin, camera is at origin, so view_z ≈ length of world_pos projected onto view forward.
    out.view_z = length(out.world_pos); // world-space distance from camera (rotation-invariant)
    out.vertex_color = in.color;
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
// Cascaded Shadow Map sampling: 9-tap Gaussian PCF (Castano '13 / The Witness / Bevy)
//
// Uses hardware comparison sampler (textureSampleCompareLevel) which provides
// free 2×2 PCF per tap. 9 bilinear taps ≈ 36 effective shadow samples.
// ---------------------------------------------------------------------------

fn csm_shadow(world_pos: vec3<f32>, view_z: f32) -> f32 {
    if config.voxel_shadows_enabled == 0u { return 1.0; }

    // Select cascade based on view-space depth.
    var cascade = 3u;
    if view_z < cascades.splits.x { cascade = 0u; }
    else if view_z < cascades.splits.y { cascade = 1u; }
    else if view_z < cascades.splits.z { cascade = 2u; }

    // Transform world position to light clip space for the selected cascade.
    let light_clip = cascades.light_vp[cascade] * vec4(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;

    // Convert from NDC [-1,1] to shadow map UV [0,1].
    let uv = vec2(light_ndc.x * 0.5 + 0.5, 1.0 - (light_ndc.y * 0.5 + 0.5));
    let depth = light_ndc.z;

    // Bail if outside the shadow map bounds.
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0 {
        return 1.0;
    }

    // 9-sample Gaussian PCF (Castano '13).
    // 9 bilinear taps approximating a 5×5 Gaussian kernel.
    // Each textureSampleCompareLevel gives hardware 2×2 PCF for free.
    let texel = 1.0 / 2048.0;

    var shadow = 0.0;
    // Offsets in texels: 3×3 grid at 1.5-texel spacing.
    shadow += 1.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2(-1.5, -1.5) * texel, cascade, depth);
    shadow += 2.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2( 0.0, -1.5) * texel, cascade, depth);
    shadow += 1.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2( 1.5, -1.5) * texel, cascade, depth);
    shadow += 2.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2(-1.5,  0.0) * texel, cascade, depth);
    shadow += 4.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2( 0.0,  0.0) * texel, cascade, depth);
    shadow += 2.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2( 1.5,  0.0) * texel, cascade, depth);
    shadow += 1.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2(-1.5,  1.5) * texel, cascade, depth);
    shadow += 2.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2( 0.0,  1.5) * texel, cascade, depth);
    shadow += 1.0 * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2( 1.5,  1.5) * texel, cascade, depth);

    return shadow / 16.0; // sum of weights = 16
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Per-vertex color modulated by uniform tint.
    let base_color = in.vertex_color * u.color.rgb;

    // Emissive blocks (light-emitting, e.g., energy crystals, lava).
    if u.color.a > 0.5 {
        return vec4(base_color, 1.0);
    }

    let metallic = u.material.x;
    let roughness = max(u.material.y, 0.04);
    let glass = u.material.z;

    // Glass material: screen-door transparency.
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

    // Cascaded shadow map sampling (9-tap Gaussian PCF).
    let shadow = csm_shadow(in.world_pos, in.view_z);

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

    var color = ambient_color + sun_contrib + block_contrib;

    // When HDR is disabled, apply tonemapping inline (LDR fallback path).
    // When HDR is enabled, output linear HDR — tonemapping happens in composite pass.
    if config.hdr_enabled == 0u {
        color = color * (color * 2.51 + vec3(0.03)) / (color * (color * 2.43 + vec3(0.59)) + vec3(0.14));
    }

    // Alpha channel encodes exterior flag for atmosphere mask.
    // material.w = 1.0 for exterior/planet chunks, 0.0 for ship interior.
    return vec4(color, u.material.w);
}
