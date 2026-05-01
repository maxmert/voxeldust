// Deferred-path eclipse — fullscreen post-process pass that runs
// after Bevy's `DeferredLightingPass` and applies the same surgical
// sun-only attenuation the forward extension applies per-fragment.
//
// Pipeline: read the radiance buffer Bevy just produced, reconstruct
// the deferred `PbrInput` from the G-buffer (matching what Bevy's
// own deferred lighting did, including the depth-prepass read that
// recovers the fragment's world position), compute the visible-sun
// fraction at this pixel via the shared `voxeldust::eclipse_lib`,
// re-derive the would-be directional-sun contribution, and SUBTRACT
// `(1 - factor) * sun_term * view.exposure` from the radiance.
//
// IBL, ambient, point + spot lights, and the cascade shadow map are
// untouched — same physical model as the forward path.
//
// Bind groups:
//   * @group(0): MeshViewBindGroup main — view, lights,
//     deferred_prepass_texture, depth_prepass_texture, etc.
//   * @group(1): MeshViewBindGroup binding_array — light probes etc.
//   * @group(2): our pass-specific bindings — radiance source, sampler,
//     eclipse uniform.

#import bevy_pbr::pbr_deferred_functions::pbr_input_from_deferred_gbuffer
#import bevy_pbr::pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT
#import bevy_pbr::mesh_view_bindings::{view, deferred_prepass_texture}
#import bevy_pbr::prepass_utils
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import voxeldust::eclipse_lib::{
    EclipseUniform,
    ECLIPSE_NO_EFFECT_THRESHOLD,
    compute_eclipse_factor,
    sun_direct_contribution,
}

@group(2) @binding(0) var radiance_source: texture_2d<f32>;
@group(2) @binding(1) var radiance_sampler: sampler;
@group(2) @binding(2) var<uniform> eclipse: EclipseUniform;

// Reverse-Z depth for the far plane / sky. Bevy clears the depth
// attachment to this value at the start of the frame, so any pixel
// where no opaque fragment wrote depth still reads as `0.0`. The
// guard skips the eclipse subtraction on those pixels — the sky's
// radiance comes from the atmosphere / starfield / clear color, not
// from `apply_pbr_lighting`, so subtracting a "would-be sun term"
// derived from cleared (zero) G-buffer data would corrupt it.
const REVERSE_Z_FAR_PLANE: f32 = 0.0;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Sample the radiance Bevy's deferred lighting just wrote.
    let radiance = textureSample(radiance_source, radiance_sampler, in.uv);

    // Read the depth this fragment landed on. `prepass_depth` reads
    // the depth-prepass texture that Bevy wrote during the deferred
    // prepass; we mirror Bevy's own deferred-lighting setup
    // (`deferred_lighting.wgsl:54-57`). Without populating
    // `frag_coord.z` from the depth texture, `pbr_input_from_deferred_gbuffer`
    // reconstructs world position at the far plane regardless of
    // where the actual surface is — silently breaking the eclipse
    // factor at every fragment.
    let depth = prepass_utils::prepass_depth(in.position, 0u);
    if (depth <= REVERSE_Z_FAR_PLANE) {
        // Sky / no-surface pixel: leave the radiance untouched.
        return radiance;
    }

    var frag_coord = vec4<f32>(in.position.xy, depth, 0.0);

    // Pull the G-buffer payload Bevy wrote during the deferred prepass.
    // `pbr_input_from_deferred_gbuffer` reconstructs world position
    // from depth (via `view_transformations::position_ndc_to_world`),
    // unpacks albedo + roughness + metallic + reflectance from the
    // RGBA8 packed channels, and decodes the octahedral normal.
    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>(frag_coord.xy), 0);
    let pbr_input = pbr_input_from_deferred_gbuffer(frag_coord, deferred_data);

    // Skip unlit fragments (sub-block emitters, decals, particle
    // billboards, etc.). The sun never directly lit them — there's
    // no sun term in the radiance to subtract.
    if ((pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u) {
        return radiance;
    }

    let eclipse_f = compute_eclipse_factor(eclipse, pbr_input.world_position.xyz);
    if (eclipse_f >= ECLIPSE_NO_EFFECT_THRESHOLD) {
        return radiance;
    }

    // Bevy's deferred lighting scales `direct_light` by `view.exposure`
    // before adding it to the radiance buffer (`pbr_functions.wgsl:745`,
    // same as the forward path). We mirror that scaling here so the
    // subtraction lands on the same magnitude as the original sun
    // contribution.
    let sun_term = sun_direct_contribution(pbr_input) * view.exposure;
    let attenuated = radiance.rgb - sun_term * (1.0 - eclipse_f);
    return vec4<f32>(max(attenuated, vec3<f32>(0.0)), radiance.a);
}
