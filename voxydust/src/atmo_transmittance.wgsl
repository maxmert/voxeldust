// Transmittance LUT compute shader. Output: 256×64 RGBA16F.
// Each texel stores the fractional transmittance from a point at altitude r,
// looking up at zenith angle mu = cos(theta), all the way to the top of the
// atmosphere. UV uses Bruneton's non-linear parameterization.
//
// This runs ONCE per planet-seed change (atmosphere parameters are static).
//
// Shared math (types, bruneton functions, medium sampling) is concatenated
// from atmo_common.wgsl at include_str! time.

@group(0) @binding(0)
var<uniform> atmo: AtmosphereUniforms;

@group(0) @binding(1)
var transmittance_out: texture_storage_2d<rgba16float, write>;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(u32(TRANSMITTANCE_LUT_W), u32(TRANSMITTANCE_LUT_H));
    if id.x >= size.x || id.y >= size.y { return; }

    let uv = (vec2<f32>(id.xy) + vec2(0.5)) / vec2<f32>(size);
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let rm = transmittance_lut_uv_to_r_mu(inner_r, outer_r, uv);
    let r = rm.x;
    let mu = rm.y;

    // Integrate extinction from (r, mu) to the top of the atmosphere.
    let t_max = distance_to_top_atmosphere(inner_r, outer_r, r, mu);
    let n = TRANSMITTANCE_SAMPLES;
    var prev_t = 0.0;
    var optical_depth = vec3(0.0);
    for (var i = 0u; i < n; i++) {
        let t = t_max * (f32(i) + MIDPOINT_RATIO) / f32(n);
        let dt = t - prev_t;
        prev_t = t;
        let r_i = step_r(r, mu, t);
        let alt = r_i - inner_r;
        let m = sample_medium(atmo, alt);
        optical_depth += m.extinction * dt;
    }

    let transmittance = exp(-optical_depth);
    textureStore(transmittance_out, vec2<i32>(id.xy), vec4(transmittance, 1.0));
}
