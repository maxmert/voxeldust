// THE PROBE (the voxel foundation, slice 8p; ruling V14 D8-7) — a material that writes a CODE
// instead of a colour, into the second picture every judged capture carries. Each pixel says WHAT
// drew it and HOW FAR it is:
//
//   R byte:  kind << 5 | rung        (the uniform's `code`, built by `vd_client_harness::probe::probe_byte`)
//   G, B:    the distance from the eye in CELLS of the rung, big-endian u16
//   A:       255
//
// ★ THE BYTES ARE EXACT THROUGH AN sRGB TARGET. The picture's target is 8-bit sRGB, and the engine
// encodes a linear fragment value on store and decodes it on the blit to the output, then encodes it
// again: byte = round(255 · enc(dec(round(255 · enc(v)) / 255))). A fragment that emits the linear
// value `dec(b / 255)` therefore stores exactly `b` — the same round trip `Color::srgb` relies on for
// every marker gate. The probe camera runs no tonemapping, no dither and no multisampling, so no
// stage between this fragment and the byte mixes two values (MEASURED by the picture gate: every
// terrain pixel's rung byte must equal the rung the flag named).
//
// The vertex stage carries THE SAME GEOMORPH AND SINK as the ground's material (`ladder_fade.wgsl`)
// and the fragment stage THE SAME FAR EDGE, so the probe reads exactly the surface the picture
// shows.
//
// The render frame's eye stands at the origin (`camera_transform`; the floating origin, every chunk
// placed against the eye), so a world position's length IS its distance from the eye — the same
// reading the ground's material and its shadow pass make.

#import bevy_pbr::{
    mesh_functions,
    forward_io::VertexOutput,
    view_transformations::position_world_to_clip,
}

struct ProbeUniform {
    // kind << 5 | rung, as a float 0..255.
    code: f32,
    // One cell of the rung, in metres.
    cell_m: f32,
    // THE CROSSFADE BANDS (slice 8 step 3): (in_lo, sink_end, out_lo, out_hi), metres from the eye.
    bands: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> probe: ProbeUniform;

struct ProbeVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(8) morph: vec3<f32>,
    @location(9) sink: vec3<f32>,
}

fn whole(d: f32) -> f32 {
    return clamp((probe.bands.w - d) / (probe.bands.w - probe.bands.z), 0.0, 1.0);
}

fn risen(d: f32) -> f32 {
    return clamp((d - probe.bands.x) / (probe.bands.y - probe.bands.x), 0.0, 1.0);
}

@vertex
fn vertex(vertex: ProbeVertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let own = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let d = length(own.xyz);
    let local = mix(vertex.morph, vertex.position, whole(d)) - vertex.sink * (1.0 - risen(d));
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    return out;
}

// The inverse of the sRGB transfer function: the linear value whose stored byte is `b / 255`.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.world_position.xyz);
    if d >= probe.bands.w {
        discard;
    }
    let cells = clamp(floor(d / probe.cell_m + 0.5), 0.0, 65535.0);
    let q = u32(cells);
    let hi = f32(q >> 8u) / 255.0;
    let lo = f32(q & 255u) / 255.0;
    return vec4<f32>(
        srgb_to_linear(probe.code / 255.0),
        srgb_to_linear(hi),
        srgb_to_linear(lo),
        1.0,
    );
}
