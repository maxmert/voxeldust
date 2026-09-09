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
// The render frame's eye stands at the origin (`camera_transform`), so a world position's length IS
// its distance from the eye; the view's own position is used all the same, so a camera that moves
// does not break the reading.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct ProbeUniform {
    // kind << 5 | rung, as a float 0..255.
    code: f32,
    // One cell of the rung, in metres.
    cell_m: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> probe: ProbeUniform;

// The inverse of the sRGB transfer function: the linear value whose stored byte is `b / 255`.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = distance(in.world_position.xyz, view.world_position);
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
