// Sphere rendering shader with uniform buffer for MVP + color.

struct Uniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4(in.position, 1.0);
    out.world_normal = in.position; // unit sphere: position IS the normal
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sun direction (hardcoded for now — will come from WorldState lighting).
    let sun_dir = normalize(vec3(0.3, 1.0, 0.5));
    let ambient = 0.1;

    // Emissive check: alpha > 0.5 means self-luminous (star).
    if u.color.a > 0.5 {
        return vec4(u.color.rgb, 1.0);
    }

    let diffuse = max(dot(normalize(in.world_normal), sun_dir), 0.0);
    let brightness = ambient + diffuse * 0.9;

    return vec4(u.color.rgb * brightness, 1.0);
}
