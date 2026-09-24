enable dual_source_blending;

#import bevy_pbr::atmosphere::{
    bindings::{view, settings},
    functions::{
        sample_transmittance_lut, sample_transmittance_lut_segment,
        sample_sky_view_lut, direction_world_to_atmosphere,
        uv_to_ray_direction, uv_to_ndc, sample_aerial_view_lut,
        sample_sun_radiance, ndc_to_camera_dist, raymarch_atmosphere, 
        get_view_position, max_atmosphere_distance, get_raymarch_segment,
        sample_density_lut, sample_local_inscattering,
        ABSORPTION_DENSITY, SCATTERING_DENSITY, MIN_EXTINCTION
    },
};

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#ifdef MULTISAMPLED
@group(0) @binding(13) var depth_texture: texture_depth_multisampled_2d;
// VOXELDUST PATCH (the air over a pixel marched once, 2026-09-24): the per-pixel march's result
// (`sky_air.wgsl`): (in-scattered light, the samples' mean distance) and (transmittance, class).
@group(0) @binding(14) var air_light: texture_2d<f32>;
@group(0) @binding(15) var air_transmittance: texture_2d<f32>;
#else
@group(0) @binding(13) var depth_texture: texture_depth_2d;
#endif

// The pixel's class (the values `sky_air.wgsl` writes).
const AIR_SKY: f32 = 1.0;
const AIR_GROUND: f32 = 2.0;

struct RenderSkyOutput {
#ifdef DUAL_SOURCE_BLENDING
    @location(0) @blend_src(0) inscattering: vec4<f32>,
    @location(0) @blend_src(1) transmittance: vec4<f32>,
#else
    @location(0) inscattering: vec4<f32>,
#endif
}

// VOXELDUST PATCH (per-sample sky): under multisampling the fragment runs ONCE PER SAMPLE (the
// `sample_index` builtin asks for it) and reads THAT sample's depth, so a multisampled edge between
// the ground and the sky keeps its coverage through this pass and the resolve blends it. Upstream
// reads sample 0 and writes one colour to every sample, which flattens the edge into a stair.
//
// VOXELDUST PATCH (the air over a pixel marched once, 2026-09-24): a sample no longer marches when
// its pixel was marched once by `sky_air.wgsl` — an all-sky pixel takes that march as it is (the
// sample's own inputs are the pixel's); an all-ground pixel takes the march to the samples' mean
// distance and extends or shortens it to its own distance by ONE PARTIAL STEP of the same march
// (the medium at the partial step's middle, the march's own step formula), so every sample keeps
// its own transmittance and its own light. Any other pixel marches per sample, as before.
#ifdef MULTISAMPLED
@fragment
fn main(in: FullscreenVertexOutput, @builtin(sample_index) sample_index: u32) -> RenderSkyOutput {
    let depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), i32(sample_index));
#else
@fragment
fn main(in: FullscreenVertexOutput) -> RenderSkyOutput {
    let depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);
#endif

    let ray_dir_ws = uv_to_ray_direction(in.uv);
    let world_pos = get_view_position();
    let r = length(world_pos);
    let up = normalize(world_pos);
    let mu = dot(ray_dir_ws, up);
    let max_samples = settings.sky_max_samples;
    let should_raymarch = settings.rendering_method == 1u;

    var transmittance: vec3<f32>;
    var inscattering: vec3<f32>;

    let sun_radiance = sample_sun_radiance(ray_dir_ws);

#ifdef MULTISAMPLED
    let air_pixel = vec2<i32>(in.position.xy);
    let air_l = textureLoad(air_light, air_pixel, 0);
    let air_t = textureLoad(air_transmittance, air_pixel, 0);
    let air_class = air_t.w;
#else
    let air_l = vec4(0.0);
    let air_t = vec4(0.0);
    let air_class = 0.0;
#endif

    // VOXELDUST PATCH (the tables the raymarch overwrites, 2026-09-24): under the raymarched mode
    // the lookup tables' values were read and then overwritten, so only the mode's own path reads.
    if depth == 0.0 {
        if should_raymarch {
            if air_class == AIR_SKY {
                inscattering = air_l.rgb;
                transmittance = air_t.rgb;
            } else {
                let t_max = max_atmosphere_distance(r, mu);
                let result = raymarch_atmosphere(world_pos, ray_dir_ws, t_max, max_samples, in.uv, true);
                inscattering = result.inscattering;
                transmittance = result.transmittance;
            }
        } else {
            let ray_dir_as = direction_world_to_atmosphere(ray_dir_ws, up);
            transmittance = sample_transmittance_lut(r, mu);
            inscattering = sample_sky_view_lut(r, ray_dir_as);
        }
        inscattering += sun_radiance * transmittance;
    } else {
        let t = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));
        if should_raymarch {
            if air_class == AIR_GROUND {
                // ONE PARTIAL STEP from the pixel's mean distance to this sample's own, inside the
                // air's segment exactly as the march clips it.
                let segment = get_raymarch_segment(r, mu);
                let seg_end = max(segment.start, segment.end);
                let t_from = clamp(air_l.w, segment.start, seg_end);
                let t_to = clamp(t, segment.start, seg_end);
                let delta = t_to - t_from;
                let mid = world_pos + ray_dir_ws * (0.5 * (t_from + t_to));
                let local_r = length(mid);
                let absorption = sample_density_lut(local_r, ABSORPTION_DENSITY);
                let scattering = sample_density_lut(local_r, SCATTERING_DENSITY);
                let extinction = absorption + scattering;
                let step_transmittance = exp(-extinction * delta);
                let local_light = sample_local_inscattering(scattering, ray_dir_ws, mid);
                inscattering = air_l.rgb
                    + air_t.rgb * (local_light - local_light * step_transmittance)
                        / max(extinction, MIN_EXTINCTION);
                transmittance = air_t.rgb * step_transmittance;
            } else {
                let result = raymarch_atmosphere(world_pos, ray_dir_ws, t, max_samples, in.uv, false);
                inscattering = result.inscattering;
                transmittance = result.transmittance;
            }
        } else {
            inscattering = sample_aerial_view_lut(in.uv, t);
            transmittance = sample_transmittance_lut_segment(r, mu, t);
        }
    }

    // exposure compensation
    inscattering *= view.exposure;
    
#ifdef DUAL_SOURCE_BLENDING
    return RenderSkyOutput(vec4(inscattering, 0.0), vec4(transmittance, 1.0));
#else
    let mean_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    return RenderSkyOutput(vec4(inscattering, mean_transmittance));
#endif
    
}
