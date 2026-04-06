// Block mesh rendering shader — PBR Cook-Torrance BRDF.
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
    light_vp: mat4x4<f32>,  // sun view-projection for shadow mapping
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

@group(1) @binding(0)
var<uniform> scene: SceneLighting;

@group(2) @binding(0)
var shadow_map: texture_depth_2d;

@group(2) @binding(1)
var shadow_sampler: sampler_comparison;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) vertex_color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4(in.position, 1.0);
    // Transform normal to world space (using model matrix upper-3x3).
    // For blocks, normals are axis-aligned and provided explicitly.
    out.world_normal = normalize((u.model * vec4(in.normal, 0.0)).xyz);
    out.world_pos = (u.model * vec4(in.position, 1.0)).xyz;
    out.vertex_color = in.color;
    return out;
}

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

    // Shadow mapping with 3x3 PCF.
    let light_clip = scene.light_vp * vec4(in.world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2(light_ndc.x * 0.5 + 0.5, -light_ndc.y * 0.5 + 0.5);
    var shadow = 1.0;
    if light_clip.w > 0.0 && shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 && shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 {
        let texel_size = 1.0 / 2048.0;
        var total = 0.0;
        for (var x = -1; x <= 1; x++) {
            for (var y = -1; y <= 1; y++) {
                let offset = vec2(f32(x), f32(y)) * texel_size;
                total += textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + offset, light_ndc.z);
            }
        }
        shadow = total / 9.0;
    }

    let lo = (diffuse + specular) * light_color * n_dot_l * shadow;

    // Ambient — slightly stronger for blocks to keep interior spaces visible.
    let ambient_color = scene.ambient.x * base_color * 0.3;

    // ACES filmic tonemapping.
    var color = ambient_color + lo;
    color = color * (color * 2.51 + vec3(0.03)) / (color * (color * 2.43 + vec3(0.59)) + vec3(0.14));

    return vec4(color, 1.0);
}
