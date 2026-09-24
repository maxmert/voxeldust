// VOXELDUST PATCH (the air over a pixel marched once, 2026-09-24): the per-pixel half of the sky.
//
// Under multisampling the sky pass runs per sample (the edge between the ground and the sky keeps
// its coverage), and every sample of a pixel marched the SAME ray — its direction comes from the
// pixel's centre — 16 steps deep, differing only where it stops. MEASURED: that march was about
// half of every window frame (lookup mode instead: 13.8 ms saved of 29 on foot, 10.2 of 22 at
// 6 376 km). This pass marches each pixel ONCE and stores what the sky pass needs:
//   * every sample sky: the sky's march itself (the samples' own inputs are this pixel's);
//   * every sample ground, the samples within ONE STEP of the march of each other: the march to
//     their MEAN distance; each sample then extends or shortens it to its own distance by one
//     partial step of the same march (`render_sky.wgsl`), so each sample keeps its own
//     transmittance and its own light, to the march's own step accuracy;
//   * anything else (an edge between the ground and the sky, a ridge over a far valley): nothing;
//     the sample marches in full, as before.
#import bevy_pbr::atmosphere::{
    bindings::settings,
    functions::{
        uv_to_ray_direction, uv_to_ndc, ndc_to_camera_dist, raymarch_atmosphere,
        get_view_position, max_atmosphere_distance, get_raymarch_segment
    },
};

@group(0) @binding(13) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(14) var air_light: texture_storage_2d<rgba32float, write>;
@group(0) @binding(15) var air_transmittance: texture_storage_2d<rgba32float, write>;

// The pixel's class, in the transmittance texture's fourth channel (read by `render_sky.wgsl`).
const AIR_PER_SAMPLE: f32 = 0.0;
const AIR_SKY: f32 = 1.0;
const AIR_GROUND: f32 = 2.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(depth_texture);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    // The pixel's centre, as the full-screen triangle hands it to the fragment.
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let ray_dir = uv_to_ray_direction(uv);
    let world_pos = get_view_position();
    let r = length(world_pos);
    let mu = dot(ray_dir, normalize(world_pos));
    let max_samples = settings.sky_max_samples;
    let ndc = uv_to_ndc(uv);

    let samples = i32(textureNumSamples(depth_texture));
    var sky = 0;
    var t_near = 3.4e38;
    var t_far = 0.0;
    var t_sum = 0.0;
    for (var s = 0; s < samples; s++) {
        let d = textureLoad(depth_texture, pixel, s);
        if d == 0.0 {
            sky += 1;
        } else {
            let t = ndc_to_camera_dist(vec3(ndc, d));
            t_near = min(t_near, t);
            t_far = max(t_far, t);
            t_sum += t;
        }
    }

    var light = vec4(0.0);
    var transmittance = vec4(1.0, 1.0, 1.0, AIR_PER_SAMPLE);
    if sky == samples {
        let result = raymarch_atmosphere(world_pos, ray_dir, max_atmosphere_distance(r, mu), max_samples, uv, true);
        light = vec4(result.inscattering, 0.0);
        transmittance = vec4(result.transmittance, AIR_SKY);
    } else if sky == 0 {
        let segment = get_raymarch_segment(r, mu);
        let air_m = min(segment.end, t_far) - segment.start;
        let steps = mix(1.0, f32(max_samples), saturate(t_far * 0.01));
        if air_m <= 0.0 || (t_far - t_near) * steps <= air_m {
            let t_mean = t_sum / f32(samples);
            let result = raymarch_atmosphere(world_pos, ray_dir, t_mean, max_samples, uv, false);
            light = vec4(result.inscattering, t_mean);
            transmittance = vec4(result.transmittance, AIR_GROUND);
        }
    }
    textureStore(air_light, pixel, light);
    textureStore(air_transmittance, pixel, transmittance);
}
