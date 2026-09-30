// ★ THE RENDER PICTURE FOR THE GATES (the temporal path's T2; `docs/investigation/2026-09-28/
// temporal_path_design.md` §12.3): the jittered picture the main passes drew, BEFORE the history, on the render
// grid, through the gate's tone map `c ÷ (1 + max c)` (the reconstruction's own reversible map, the tone weight of
// `vd_client::reconstruction`), written to the capture's eight-bit sRGB image beside the probe.
//
// The render picture fills the top-left render rectangle of the view's full-size texture, and the image is the
// render grid, so a pixel reads the texture at its own place.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var source: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let c = max(textureLoad(source, vec2<i32>(in.position.xy), 0).rgb, vec3<f32>(0.0));
    return vec4<f32>(c / (max(c.r, max(c.g, c.b)) + 1.0), 1.0);
}
