// Aerial-perspective LUT compute shader. Output: 32×32×32 RGBA16F (3D).
// Each texel stores (inscatter.rgb, mean_transmittance) for the view frustum
// voxel defined by (uv.xy, depth_slice). uv covers the screen; depth slice is
// a fraction of aerial_view_lut_max_distance (km).
//
// Samples are log-encoded in the inscatter channels so linear interpolation
// across slices preserves the exponential nature of light accumulation.
//
// Rebuilt EVERY FRAME — depends on camera orientation.
//
// Shared math concatenated from atmo_common.wgsl at include_str! time.

@group(0) @binding(0)
var<uniform> atmo: AtmosphereUniforms;

@group(0) @binding(1)
var transmittance_lut: texture_2d<f32>;

@group(0) @binding(2)
var lut_sampler: sampler;

@group(0) @binding(3)
var multiscatter_lut: texture_2d<f32>;

@group(0) @binding(4)
var aerial_out: texture_storage_3d<rgba16float, write>;

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let uv = transmittance_lut_r_mu_to_uv(inner_r, outer_r, r, mu);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

fn sample_multiscatter(r: f32, mu: f32) -> vec3<f32> {
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let uv = multiscatter_r_mu_to_uv(inner_r, outer_r, r, mu);
    return textureSampleLevel(multiscatter_lut, lut_sampler, uv, 0.0).rgb;
}

// Reconstruct a world-space ray direction from screen UV using inv_vp.
fn uv_to_ray_ws(uv: vec2<f32>) -> vec3<f32> {
    // Reverse-Z convention: near plane is depth = 1. Pick a point on the near
    // plane so inv_vp yields a meaningful direction even with infinite far.
    let ndc = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let world_h = atmo.inv_vp * ndc;
    let world_p = world_h.xyz / world_h.w;
    return normalize(world_p);
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(u32(AERIAL_LUT_W), u32(AERIAL_LUT_H));
    if id.x >= size.x || id.y >= size.y { return; }

    let uv = (vec2<f32>(id.xy) + vec2(0.5)) / vec2<f32>(size);
    let ray_ws = uv_to_ray_ws(uv);

    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let observer_ws = atmo.observer_pos.xyz;
    // Match sky-view clamping so both LUTs agree on the effective observer.
    let r_obs = clamp(length(observer_ws), inner_r + 0.001, outer_r - 0.001);

    let t_max = max(atmo.quality.z, 1.0);
    let num_slices = u32(AERIAL_LUT_D);
    let samples_per_slice = AERIAL_SAMPLES_PER_SLICE;

    let sun_dir_ws = atmo.sun_dir.xyz;
    let sun_illum = atmo.sun_color.xyz * max(atmo.quality.y, 0.0);
    let g = atmo.mie.w;
    let cos_theta = dot(ray_ws, sun_dir_ws);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, g);

    var prev_t = 0.0;
    var inscatter = vec3(0.0);
    var transmittance = vec3(1.0);

    for (var slice_i = 0u; slice_i < num_slices; slice_i++) {
        let slice_end = t_max * (f32(slice_i) + 1.0) / f32(num_slices);

        for (var step_i = 0u; step_i < samples_per_slice; step_i++) {
            let t = t_max * (f32(slice_i) + (f32(step_i) + MIDPOINT_RATIO) / f32(samples_per_slice)) / f32(num_slices);
            let dt = t - prev_t;
            prev_t = t;

            let sample_ws = observer_ws + ray_ws * t;
            let r_i = max(length(sample_ws), inner_r + 0.001);
            let alt = r_i - inner_r;
            let m = sample_medium(atmo, alt);
            let seg_transmittance = exp(-m.extinction * dt);

            let up_ws = normalize(sample_ws);
            let mu_sun = dot(up_ws, sun_dir_ws);
            let hits = ray_hits_ground(inner_r, r_i, mu_sun);
            var trans_to_sun = vec3(0.0);
            if !hits {
                trans_to_sun = sample_transmittance(r_i, mu_sun);
            }

            let scattered = m.rayleigh_scattering * phase_r + vec3(m.mie_scattering) * phase_m;
            let scattering_iso = m.rayleigh_scattering + vec3(m.mie_scattering);
            let psi_ms = sample_multiscatter(r_i, mu_sun);
            let inscat_local = sun_illum * (trans_to_sun * scattered + psi_ms * scattering_iso);
            let s_int = (inscat_local - inscat_local * seg_transmittance) / max(m.extinction, MIN_EXTINCTION);
            inscatter += transmittance * s_int;
            transmittance *= seg_transmittance;
        }

        // Store log-encoded inscatter in RGB. Alpha = mean transmittance.
        let log_inscatter = log(max(inscatter, vec3(1e-6)));
        let mean_t = (transmittance.r + transmittance.g + transmittance.b) * (1.0 / 3.0);
        textureStore(
            aerial_out,
            vec3<i32>(i32(id.x), i32(id.y), i32(slice_i)),
            vec4(log_inscatter, mean_t),
        );

        if all(transmittance < vec3(0.001)) {
            // Fill remaining slices with the same final value.
            for (var remain = slice_i + 1u; remain < num_slices; remain++) {
                textureStore(
                    aerial_out,
                    vec3<i32>(i32(id.x), i32(id.y), i32(remain)),
                    vec4(log_inscatter, mean_t),
                );
            }
            break;
        }
    }
}
