// Shared eclipse math + types, imported by both the forward
// material-extension shader (`eclipse.wgsl`) and the deferred
// post-process pass (`eclipse_deferred.wgsl`).
//
// Two top-level operations are exported:
//
//   * `compute_eclipse_factor(eclipse, world_position)` —
//     visible-sun fraction at `world_position`, accounting for every
//     active occluder in the supplied uniform. Soft penumbra via
//     angular-disk lens-area intersection; smooth ramp through 1st-
//     contact / annular / total / 4th-contact.
//
//   * `sun_direct_contribution(pbr_input)` — re-derives the
//     directional sun's contribution at this fragment, mirroring the
//     setup `apply_pbr_lighting` performs internally (and returning
//     the same value Bevy's lighting loop would have added). The
//     forward extension subtracts `(1 - factor) * this * view.exposure`
//     from the lit color; the deferred pass does the same against
//     the radiance buffer.
//
// Pass-by-value of `EclipseUniform` (160 bytes) keeps the function
// signatures decoupled from the calling shader's bind-group layout —
// each shader binds the uniform at whatever group/binding suits its
// pipeline, then passes it in by value. Compilers fold the struct
// load into register reads with no observable cost on any modern GPU.
//
// **Bevy upgrade tracking**: `sun_direct_contribution` mirrors the
// `LightingInput` setup in `bevy_pbr/src/render/pbr_functions.wgsl::
// apply_pbr_lighting`. If Bevy changes that signature or adds new
// `LightingInput` fields, the subtraction will diverge from the
// actual sun term. Verify on each Bevy bump. Today's mirror targets
// bevy_pbr-0.18.1.

#define_import_path voxeldust::eclipse_lib

#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0}
#import bevy_pbr::lighting
#import bevy_pbr::pbr_types::PbrInput
#import bevy_pbr::shadows
#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_pbr::mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT
#import bevy_pbr::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT

const ECLIPSE_MAX_OCCLUDERS: u32 = 8u;
const ECLIPSE_PI: f32 = 3.141592653589793;

// Below this visible-sun fraction we treat the eclipse as observable
// and run the per-fragment sun-recompute. Above it the eclipse is a
// no-op and the per-fragment recompute is skipped — bounds the worst-
// case cost on the >99 % of pixels not in any penumbra.
const ECLIPSE_NO_EFFECT_THRESHOLD: f32 = 0.9995;

// Distances under this metre cap are treated as "fragment coincident
// with body" — guards against division-by-zero when the camera (and
// hence a chunk fragment) lands at body-centre coordinates.
const ECLIPSE_NEAR_ZERO_M: f32 = 1.0;

struct EclipseUniform {
    // .xyz = sun position camera-relative (m); .w = sun radius (m).
    // `w == 0` is the sentinel for "no sun configured" — every entry
    // point short-circuits to a no-op when this is the case.
    sun: vec4<f32>,
    // .xyz = occluder position camera-relative (m); .w = radius (m).
    // Inactive slots have `w == 0` and are skipped.
    occluders: array<vec4<f32>, 8>,
    // .x = active occluder count (≤ ECLIPSE_MAX_OCCLUDERS); the rest
    // is std140 padding.
    count: vec4<u32>,
}

// Lens-area intersection of two angular disks (Sun + occluder),
// returning the visible fraction of the sun's disk in [0, 1].
// Inputs in radians.
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
    let sun_area = ECLIPSE_PI * r1_sq;
    return clamp(1.0 - lens_area / sun_area, 0.0, 1.0);
}

// Visible-sun fraction at `world_position`, accounting for every
// active occluder broadcast in `eclipse.occluders`. Returns 1.0
// (no occlusion) when no sun is configured.
fn compute_eclipse_factor(eclipse: EclipseUniform, world_position: vec3<f32>) -> f32 {
    if (eclipse.sun.w <= 0.0) {
        return 1.0;
    }
    let sun_pos = eclipse.sun.xyz;
    let sun_radius = eclipse.sun.w;
    let to_sun = sun_pos - world_position;
    let dist_sun = length(to_sun);
    if (dist_sun < ECLIPSE_NEAR_ZERO_M) {
        return 1.0;
    }
    let dir_sun = to_sun / dist_sun;
    let sun_ang_radius = sun_radius / dist_sun;

    var factor: f32 = 1.0;
    let n = min(eclipse.count.x, ECLIPSE_MAX_OCCLUDERS);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let occ = eclipse.occluders[i];
        if (occ.w <= 0.0) {
            continue;
        }
        let to_occ = occ.xyz - world_position;
        let dist_occ = length(to_occ);
        if (dist_occ < ECLIPSE_NEAR_ZERO_M) {
            continue;
        }
        // The occluder must be in front of the sun; bodies behind it
        // can't eclipse it (other side of the sight-line).
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

// Re-derive the would-be directional-sun contribution at this fragment.
// Mirrors `apply_pbr_lighting`'s setup of `LightingInput` and its
// directional-light loop body — keep the two in lock-step on Bevy
// upgrades. Returns the post-shadow, pre-exposure contribution; the
// caller is responsible for multiplying by `view.exposure` to match
// the magnitude of the same term inside the radiance buffer.
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

    // Directional sun is light index 0 by convention (`solar.rs` spawns
    // exactly one). If Bevy ever indexes them differently, audit
    // here — we want the system's single bright star.
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
