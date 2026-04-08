// HDR composite pass: reads HDR scene texture, applies tonemapping, outputs to LDR swapchain.
// Future: atmospheric scattering and god rays are composited here before tonemapping.

const PI: f32 = 3.14159265358979;

struct CompositeParams {
    screen_dims: vec4<f32>,  // xy = width, height; zw = 1/width, 1/height
}

@group(0) @binding(0)
var hdr_scene: texture_2d<f32>;

@group(0) @binding(1)
var screen_sampler: sampler;

@group(0) @binding(2)
var<uniform> params: CompositeParams;

// Vertexless fullscreen triangle: generates a triangle covering clip space [-1,1].
// Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3).
// The GPU clips the oversized triangle to the viewport automatically.

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
    var out: FullscreenOutput;
    // Generate oversized triangle covering entire screen.
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.position = vec4(x, y, 0.0, 1.0);
    // UV: map clip [-1,1] to texture [0,1], flip Y for wgpu convention.
    out.uv = vec2(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ACES filmic tonemapping operator.
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color * (color * 2.51 + vec3(0.03)) / (color * (color * 2.43 + vec3(0.59)) + vec3(0.14));
}

@fragment
fn fs_composite(in: FullscreenOutput) -> @location(0) vec4<f32> {
    // Sample HDR scene.
    let hdr_color = textureSample(hdr_scene, screen_sampler, in.uv).rgb;

    // TODO Phase 3: atmospheric scattering compositing here (before tonemap).
    // TODO Phase 4: god ray addition here (before tonemap).
    var color = hdr_color;

    // Tonemapping: ACES filmic.
    color = aces_tonemap(color);

    return vec4(color, 1.0);
}
