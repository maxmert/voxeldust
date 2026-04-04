// Star field rendering: instanced billboarded quads with glow and warp streaks.

struct StarSceneUniforms {
    view_proj: mat4x4<f32>,
    camera_right: vec4<f32>,  // billboard axis
    camera_up: vec4<f32>,     // billboard axis
    warp_velocity: vec4<f32>, // xyz = normalized dir, w = speed (GU/s)
    render_mode: vec4<f32>,   // x = 0.0 skybox / 1.0 galaxy, y = streak_factor, zw = unused
}

@group(0) @binding(0)
var<uniform> scene: StarSceneUniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,     // per-vertex: corner [-1,1]
    @location(1) star_pos: vec4<f32>,     // per-instance: xyz = dir/pos, w = apparent_size
    @location(2) star_color: vec4<f32>,   // per-instance: rgb = color, a = brightness
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let mode = scene.render_mode.x;
    let streak_factor = scene.render_mode.y;
    let warp_speed = scene.warp_velocity.w;

    // Star world position.
    var world_pos: vec3<f32>;
    if mode < 0.5 {
        // Skybox: direction only, placed on a large sphere.
        world_pos = normalize(in.star_pos.xyz) * 500.0;
    } else {
        // Galaxy: actual position relative to camera.
        world_pos = in.star_pos.xyz;
    }

    let apparent_size = in.star_pos.w;
    let right = scene.camera_right.xyz;
    let up = scene.camera_up.xyz;

    // Expand billboard quad.
    var offset: vec3<f32>;
    if warp_speed > 0.5 && streak_factor > 0.0 {
        // Warp streak: elongate along velocity direction.
        let vel_dir = scene.warp_velocity.xyz;
        let stretch = 1.0 + streak_factor * min(warp_speed / 10.0, 5.0);
        // Project velocity onto the billboard plane.
        let streak_right = normalize(vel_dir - dot(vel_dir, up) * up);
        offset = (in.quad_pos.x * stretch * streak_right
                + in.quad_pos.y * up) * apparent_size;
    } else {
        offset = (in.quad_pos.x * right + in.quad_pos.y * up) * apparent_size;
    }

    let billboard_pos = world_pos + offset;
    out.clip_position = scene.view_proj * vec4(billboard_pos, 1.0);
    out.uv = in.quad_pos;
    out.color = in.star_color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Radial distance from quad center.
    let r = length(in.uv);

    // Compact glow: visible core with small halo.
    let core_glow = exp(-r * r * 12.0);
    let halo = 0.15 / (1.0 + r * r * 8.0);
    let glow = min(core_glow + halo, 1.0);

    // Scale for additive blending — bright star cores are visible while
    // accumulated halos in the galactic plane create a Milky Way band.
    let alpha = glow * in.color.a * 0.2;

    if alpha < 0.002 { discard; }

    return vec4(in.color.rgb * alpha, alpha);
}

// --- Point rendering: 1 pixel per star, no billboard, no overdraw ---

struct PointVertexInput {
    @location(0) star_pos: vec4<f32>,    // xyz = direction, w = unused
    @location(1) star_color: vec4<f32>,  // rgb = color, a = brightness
}

struct PointVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_point(in: PointVertexInput) -> PointVertexOutput {
    var out: PointVertexOutput;
    let world_pos = normalize(in.star_pos.xyz) * 500.0;
    out.clip_position = scene.view_proj * vec4(world_pos, 1.0);
    out.color = in.star_color;
    return out;
}

@fragment
fn fs_point(in: PointVertexOutput) -> @location(0) vec4<f32> {
    // sqrt brightness curve: expands dim-to-mid range for visibility
    // on high-DPI displays where single pixels are physically tiny.
    let b = sqrt(in.color.a);
    let rgb = in.color.rgb * b + vec3(b * 0.1);
    return vec4(min(rgb, vec3(1.0)), 1.0);
}
