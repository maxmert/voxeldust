// Eclipse — geometric occlusion of the directional sun by celestial
// bodies broadcast via `WorldState.bodies`.
//
// Replaces Bevy's standard PBR fragment with a copy that runs the
// usual `apply_pbr_lighting`, then SUBTRACTS the would-be directional-
// sun contribution in proportion to the eclipse factor at this
// fragment. IBL, ambient, point + spot lights, and the cascade
// shadow map are untouched — eclipse darkens only what the *sun*
// directly illuminates, matching the physical effect (the sky stays
// lit by scattered + IBL light during a partial eclipse, only the
// solar disk loses energy).
//
// The eclipse term itself is a soft penumbra: angular-disk lens-area
// intersection (`visible_sun_fraction`), not a hard ray-vs-sphere
// binary. The result is a continuous [0, 1] visible-sun fraction that
// smoothly ramps through 1st-contact → totality → 4th-contact, just
// like a real eclipse.
//
// **Bevy upgrade tracking**: `sun_direct_contribution` re-derives the
// `LightingInput` for the directional sun using the same setup
// `apply_pbr_lighting` performs. If Bevy changes that signature or
// adds new fields, the subtraction will diverge from the actual sun
// term. Verify against `bevy_pbr/src/render/pbr_functions.wgsl`
// (`apply_pbr_lighting`, ~line 282) on each Bevy bump. Today's mirror
// targets bevy_pbr-0.18.1.

#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::pbr_functions::{
    alpha_discard,
    apply_pbr_lighting,
    main_pass_post_lighting_processing,
    calculate_diffuse_color,
    calculate_F0,
}
#import bevy_pbr::lighting
#import bevy_pbr::pbr_types::PbrInput
#import bevy_pbr::shadows
#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_pbr::mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT
#import bevy_pbr::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT

#ifdef PREPASS_PIPELINE
    #import bevy_pbr::prepass_io::{VertexOutput, FragmentOutput}
    #import bevy_pbr::pbr_deferred_functions::deferred_output
#else
    #import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}
#endif

const MAX_OCCLUDERS: u32 = 8u;
const PI: f32 = 3.141592653589793;
// Below this visible-sun fraction we treat the eclipse as "no
// observable effect" and skip the subtraction. Removes per-fragment
// directional-light recompute on the >99 % of pixels not in any
// penumbra.
const ECLIPSE_NO_EFFECT_THRESHOLD: f32 = 0.9995;
// Distances under this metre cap are treated as "fragment coincident
// with body" — guards against division-by-zero when the camera (and
// hence a chunk fragment) lands at body-centre coordinates.
const NEAR_ZERO_DISTANCE_M: f32 = 1.0;

struct EclipseUniform {
    // .xyz = sun position camera-relative (m); .w = sun radius (m).
    sun: vec4<f32>,
    // .xyz = occluder position camera-relative (m); .w = radius (m).
    // Inactive slots have w == 0 and are skipped in the loop.
    occluders: array<vec4<f32>, 8>,
    // .x = active occluder count; rest is std140 padding.
    count: vec4<u32>,
}

// `MATERIAL_BIND_GROUP` is a Bevy preprocessor variable (= 3 in
// Bevy 0.18, see `bevy_pbr::material::MATERIAL_BIND_GROUP_INDEX`)
// that resolves to the bind group hosting the active material's
// bindings. Hard-coding `@group(N)` would wedge the shader to a
// specific Bevy version; the preprocessor variable insulates us.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> eclipse: EclipseUniform;

// Lens-area intersection of two angular disks (the Sun and an
// occluder), returning the visible fraction of the sun's disk in the
// range [0, 1]. Inputs are in radians (sun + occluder angular radii,
// and the angular separation between their centres).
fn visible_sun_fraction(r_sun: f32, r_occ: f32, theta: f32) -> f32 {
    if (theta >= r_sun + r_occ) {
        // Disks fully separated.
        return 1.0;
    }
    if (theta + r_sun <= r_occ) {
        // Sun fully inside occluder's disk → total eclipse.
        return 0.0;
    }
    if (theta + r_occ <= r_sun) {
        // Occluder fully inside sun's disk → annular eclipse.
        // Visible fraction = 1 − (r_occ / r_sun)².
        let ratio = r_occ / r_sun;
        return 1.0 - ratio * ratio;
    }
    // Partial: closed-form lens area between two circles.
    // <https://mathworld.wolfram.com/Lens.html>
    let r1_sq = r_sun * r_sun;
    let r2_sq = r_occ * r_occ;
    let d = max(theta, 1e-12);
    let a1 = clamp((d * d + r1_sq - r2_sq) / (2.0 * d * r_sun), -1.0, 1.0);
    let a2 = clamp((d * d + r2_sq - r1_sq) / (2.0 * d * r_occ), -1.0, 1.0);
    let part1 = r1_sq * acos(a1);
    let part2 = r2_sq * acos(a2);
    let factor =
        (-d + r_sun + r_occ) *
        ( d + r_sun - r_occ) *
        ( d - r_sun + r_occ) *
        ( d + r_sun + r_occ);
    let triangle = 0.5 * sqrt(max(0.0, factor));
    let lens_area = part1 + part2 - triangle;
    let sun_area = PI * r1_sq;
    return clamp(1.0 - lens_area / sun_area, 0.0, 1.0);
}

// Visible-sun fraction at `world_position`, accounting for every
// active occluder broadcast in `eclipse.occluders`. Returns 1.0 if no
// sun is configured (uniform left at INACTIVE).
fn compute_eclipse_factor(world_position: vec3<f32>) -> f32 {
    if (eclipse.sun.w <= 0.0) {
        return 1.0;
    }
    let sun_pos = eclipse.sun.xyz;
    let sun_radius = eclipse.sun.w;
    let to_sun = sun_pos - world_position;
    let dist_sun = length(to_sun);
    if (dist_sun < NEAR_ZERO_DISTANCE_M) {
        return 1.0;
    }
    let dir_sun = to_sun / dist_sun;
    let sun_ang_radius = sun_radius / dist_sun;

    var factor: f32 = 1.0;
    let n = min(eclipse.count.x, MAX_OCCLUDERS);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let occ = eclipse.occluders[i];
        if (occ.w <= 0.0) {
            continue;
        }
        let to_occ = occ.xyz - world_position;
        let dist_occ = length(to_occ);
        if (dist_occ < NEAR_ZERO_DISTANCE_M) {
            continue;
        }
        // The occluder must be in front of the sun; bodies behind the
        // sun cannot eclipse it (they're on the other side of the
        // observer's sight-line to it).
        if (dist_occ >= dist_sun) {
            continue;
        }
        let dir_occ = to_occ / dist_occ;
        let cos_theta = clamp(dot(dir_sun, dir_occ), -1.0, 1.0);
        let theta = acos(cos_theta);
        let occ_ang_radius = occ.w / dist_occ;
        let visible = visible_sun_fraction(sun_ang_radius, occ_ang_radius, theta);
        factor = factor * visible;
    }
    return factor;
}

// Re-derive the would-be directional-sun contribution at this
// fragment. Mirrors `apply_pbr_lighting`'s setup of `LightingInput`
// and its directional-light loop body — keep the two in lock-step on
// Bevy upgrades.
fn sun_direct_contribution(pbr_input: PbrInput) -> vec3<f32> {
    if (view_bindings::lights.n_directional_lights == 0u) {
        return vec3<f32>(0.0);
    }

    let perceptual_roughness = pbr_input.material.perceptual_roughness;
    let roughness = lighting::perceptualRoughnessToRoughness(perceptual_roughness);
    let metallic = pbr_input.material.metallic;
    let reflectance = pbr_input.material.reflectance;
    let diffuse_transmission = pbr_input.material.diffuse_transmission;
    let specular_transmission = pbr_input.material.specular_transmission;

    let NdotV = max(dot(pbr_input.N, pbr_input.V), 0.0001);
    let R = reflect(-pbr_input.V, pbr_input.N);

    let diffuse_color = calculate_diffuse_color(
        pbr_input.material.base_color.rgb,
        metallic,
        specular_transmission,
        diffuse_transmission,
    );
    let F0 = calculate_F0(pbr_input.material.base_color.rgb, metallic, reflectance);
    let F_ab = lighting::F_AB(perceptual_roughness, NdotV);

    var lighting_input: lighting::LightingInput;
    lighting_input.layers[lighting::LAYER_BASE].NdotV = NdotV;
    lighting_input.layers[lighting::LAYER_BASE].N = pbr_input.N;
    lighting_input.layers[lighting::LAYER_BASE].R = R;
    lighting_input.layers[lighting::LAYER_BASE].perceptual_roughness = perceptual_roughness;
    lighting_input.layers[lighting::LAYER_BASE].roughness = roughness;
    lighting_input.P = pbr_input.world_position.xyz;
    lighting_input.V = pbr_input.V;
    lighting_input.diffuse_color = diffuse_color;
    lighting_input.F0_ = F0;
    lighting_input.F_ab = F_ab;

    // Directional sun is light index 0 (`solar.rs` spawns exactly
    // one). If multiple ever land in `lights.directional_lights`, the
    // first-broadcast star always sits at index 0 — see
    // `solar.rs::update_solar_light`.
    let light_contrib = lighting::directional_light(0u, &lighting_input, true);

    var shadow: f32 = 1.0;
    if ((pbr_input.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
        && (view_bindings::lights.directional_lights[0].flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
        let view_z = dot(vec4<f32>(
            view_bindings::view.view_from_world[0].z,
            view_bindings::view.view_from_world[1].z,
            view_bindings::view.view_from_world[2].z,
            view_bindings::view.view_from_world[3].z,
        ), pbr_input.world_position);
        shadow = shadows::fetch_directional_shadow(0u, pbr_input.world_position, pbr_input.world_normal, view_z);
    }

    return light_contrib * shadow;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    // Deferred prepass: write G-buffer unchanged. The deferred
    // lighting pass that resolves these fragments runs in a separate
    // fullscreen shader and does not currently apply the eclipse
    // term — High/Ultra presets (which enable SSR → DeferredPrepass)
    // therefore see only the directional-light's photometric
    // illuminance with no eclipse modulation. A follow-up will add a
    // post-process pass that reads `EclipseUniform` and modulates
    // the radiance buffer in deferred mode. Forward-mode pipelines
    // (Low / Medium presets) get full per-fragment eclipse here.
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);

    let eclipse_f = compute_eclipse_factor(pbr_input.world_position.xyz);
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
