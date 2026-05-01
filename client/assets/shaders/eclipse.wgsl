// Forward eclipse extension — overrides Bevy's standard PBR fragment
// for `ExtendedMaterial<StandardMaterial, EclipseExt>`. Runs the
// usual `apply_pbr_lighting` to get the lit color, then SUBTRACTS
// the would-be directional-sun contribution scaled by
// `1 - eclipse_factor`. IBL, ambient, point + spot lights, and the
// cascade shadow map are untouched.
//
// All eclipse math + the `LightingInput` reconstruction live in the
// shared `voxeldust::eclipse_lib` module — the deferred post-process
// pass (`eclipse_deferred.wgsl`) imports the same module so both
// render paths produce visually identical results.

#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::pbr_functions::{
    alpha_discard,
    apply_pbr_lighting,
    main_pass_post_lighting_processing,
}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import voxeldust::eclipse_lib::{
    EclipseUniform,
    ECLIPSE_NO_EFFECT_THRESHOLD,
    compute_eclipse_factor,
    sun_direct_contribution,
}

#ifdef PREPASS_PIPELINE
    #import bevy_pbr::prepass_io::{VertexOutput, FragmentOutput}
    #import bevy_pbr::pbr_deferred_functions::deferred_output
#else
    #import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}
#endif

// `MATERIAL_BIND_GROUP` is a Bevy preprocessor variable (= 3 in
// Bevy 0.18, see `bevy_pbr::material::MATERIAL_BIND_GROUP_INDEX`)
// that resolves to the bind group hosting the active material's
// bindings. Hard-coding `@group(N)` would wedge the shader to a
// specific Bevy version; the preprocessor variable insulates us.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> eclipse: EclipseUniform;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    // Deferred prepass: write G-buffer unchanged. Per-fragment
    // eclipse for the deferred lighting pass is handled by the
    // separate post-process pass (`eclipse_deferred.wgsl`).
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);

    let eclipse_f = compute_eclipse_factor(eclipse, pbr_input.world_position.xyz);
    if (eclipse_f < ECLIPSE_NO_EFFECT_THRESHOLD) {
        // `apply_pbr_lighting` scales `direct_light` by
        // `view.exposure` before adding it to `output_color`
        // (`pbr_functions.wgsl:745`). We mirror that scaling here so
        // our subtraction lands on the same magnitude as the original
        // sun contribution.
        let sun_term = sun_direct_contribution(pbr_input) * view_bindings::view.exposure;
        let attenuated = out.color.rgb - sun_term * (1.0 - eclipse_f);
        out.color = vec4<f32>(max(attenuated, vec3<f32>(0.0)), out.color.a);
    }

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
