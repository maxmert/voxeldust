// Multi-scattering LUT compute shader. Output: 32×32 RGBA16F.
// Each texel stores ψ_ms(r, μ_sun) — the infinite-order multi-scattering
// contribution per unit scattering at altitude r, sun zenith cos μ_sun.
// Sampled by sky-view / aerial-view shaders as an additive isotropic boost.
//
// Algorithm: Hillaire 2020 §5.2 / Bevy port.
//   - 64 rays on a Lambert equal-area sphere per (r, μ) texel.
//   - Each ray integrates single-scatter L₂ + in-medium f_ms.
//   - Parallel reduction across workgroup (64 threads).
//   - Geometric series closed form: ψ_ms = L₂ / (1 - f_ms).
//
// One workgroup per output texel: workgroup_size (1, 1, 64).
//
// Shared math concatenated from atmo_common.wgsl at include_str! time.

@group(0) @binding(0)
var<uniform> atmo: AtmosphereUniforms;

@group(0) @binding(1)
var transmittance_lut: texture_2d<f32>;

@group(0) @binding(2)
var lut_sampler: sampler;

@group(0) @binding(3)
var multiscatter_out: texture_storage_2d<rgba16float, write>;

const FRAC_4_PI: f32 = 0.07957747154594767;
const MULTISCATTER_LUT_W: f32 = 32.0;
const MULTISCATTER_LUT_H: f32 = 32.0;
const MULTISCATTER_SAMPLES: u32 = 20u;

// Plastic-number quasirandom sequence for stratified sphere sampling.
const PHI_2: vec2<f32> = vec2(1.3247179572447460259609088, 1.7548776662466927600495087);

fn s2_sequence(n: u32) -> vec2<f32> {
    return fract(vec2(0.5) + f32(n) * PHI_2);
}

// Lambert equal-area projection of unit square → unit sphere.
fn uv_to_sphere(uv: vec2<f32>) -> vec3<f32> {
    let phi = PI_2 * uv.y;
    let sin_lambda = 2.0 * uv.x - 1.0;
    let cos_lambda = sqrt(max(1.0 - sin_lambda * sin_lambda, 0.0));
    return vec3(cos_lambda * cos(phi), cos_lambda * sin(phi), sin_lambda);
}

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let uv = transmittance_lut_r_mu_to_uv(inner_r, outer_r, r, mu);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

// Decode (r, μ) from the multi-scatter LUT's UV parameterization.
// Bevy's parameterization: u = 0.5 + 0.5*mu (μ ∈ [-1, 1]), v = (r - inner) / (outer - inner).
fn multiscatter_uv_to_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let size = vec2(MULTISCATTER_LUT_W, MULTISCATTER_LUT_H);
    let adj = sub_uvs_to_unit(uv, size);
    let r = mix(inner_r, outer_r, adj.y);
    let mu = adj.x * 2.0 - 1.0;
    return vec2(r, mu);
}

struct MultiscatterSample {
    l_2: vec3<f32>,
    f_ms: vec3<f32>,
}

fn sample_multiscatter_dir(r: f32, ray_dir: vec3<f32>, light_dir: vec3<f32>) -> MultiscatterSample {
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;

    // Atmosphere-space convention: local up is +Y; ray_dir.y = cos(zenith).
    let mu_view = ray_dir.y;
    let t_max = max_atmosphere_distance(inner_r, outer_r, r, mu_view);
    let n = MULTISCATTER_SAMPLES;
    let dt = t_max / f32(n);

    var l_2 = vec3(0.0);
    var f_ms = vec3(0.0);
    var throughput = vec3(1.0);
    var optical_depth = vec3(0.0);

    for (var i = 0u; i < n; i++) {
        let t = dt * (f32(i) + 0.5);
        let local_r = step_r(r, mu_view, t);
        // Local-up at sample point (atmosphere space, observer at (0, r, 0)).
        let sample_pos = vec3(0.0, r, 0.0) + ray_dir * t;
        let local_up = normalize(sample_pos);
        let alt = local_r - inner_r;
        let m = sample_medium(atmo, alt);
        let scattering = m.rayleigh_scattering + vec3(m.mie_scattering);
        let extinction = m.extinction;
        let sample_transmittance = exp(-extinction * dt);
        optical_depth += extinction * dt;

        // In-medium multi-scatter integrand (isotropic).
        let ms_int = (scattering - scattering * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        f_ms += throughput * ms_int;

        // Single-scatter from directional light (sun), isotropic phase.
        let mu_light = dot(light_dir, local_up);
        var shadow_factor = vec3(0.0);
        if !ray_hits_ground(inner_r, local_r, mu_light) {
            shadow_factor = sample_transmittance(local_r, mu_light);
        }
        let s = scattering * shadow_factor * FRAC_4_PI;
        let s_int = (s - s * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        l_2 += throughput * s_int;

        throughput *= sample_transmittance;
        if all(throughput < vec3(0.001)) { break; }
    }

    // Ground reflection (Lambertian) — contributes to multi-scatter if the
    // ray hits the surface.
    if ray_hits_ground(inner_r, r, mu_view) {
        let transmittance_to_ground = exp(-optical_depth);
        let t_g = distance_to_ground(inner_r, r, mu_view);
        let sample_pos = vec3(0.0, r, 0.0) + ray_dir * t_g;
        let local_up = normalize(sample_pos);
        let mu_light = dot(light_dir, local_up);
        let transmittance_to_light = sample_transmittance(inner_r, mu_light);
        let albedo = vec3(atmo.ozone_extra.y);
        let ground_lum = transmittance_to_light * transmittance_to_ground * max(mu_light, 0.0) * albedo;
        l_2 += ground_lum;
    }

    return MultiscatterSample(l_2, f_ms);
}

var<workgroup> ms_shared: array<vec3<f32>, 64>;
var<workgroup> l2_shared: array<vec3<f32>, 64>;

@compute
@workgroup_size(1, 1, 64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = (vec2<f32>(id.xy) + vec2(0.5)) / vec2(MULTISCATTER_LUT_W, MULTISCATTER_LUT_H);
    let r_mu = multiscatter_uv_to_r_mu(uv);
    let r = r_mu.x;
    let mu_sun = r_mu.y;
    // Light points "from the +Y direction of mu_sun" — Bevy uses normalize(vec3(0, mu_sun, -1)).
    // Direction vector TO light (conventional): components give mu_sun against local up.
    let light_dir = normalize(vec3(0.0, mu_sun, -1.0));

    let ray_dir = uv_to_sphere(s2_sequence(id.z));
    let s = sample_multiscatter_dir(r, ray_dir, light_dir);

    // Uniform weights across the sphere.
    let sphere_solid_angle = 4.0 * PI;
    let weight = sphere_solid_angle / 64.0;
    ms_shared[id.z] = s.f_ms * weight;
    l2_shared[id.z] = s.l_2 * weight;

    workgroupBarrier();

    // Parallel reduction.
    for (var step = 32u; step > 0u; step = step >> 1u) {
        if id.z < step {
            ms_shared[id.z] = ms_shared[id.z] + ms_shared[id.z + step];
            l2_shared[id.z] = l2_shared[id.z] + l2_shared[id.z + step];
        }
        workgroupBarrier();
    }

    if id.z > 0u { return; }

    // Isotropic phase normalization + infinite-order geometric series.
    // Clamp the denominator aggressively: f_ms → 1 is a degenerate "all energy
    // stays in the medium" limit which produces visually divergent ψ_ms. The
    // 0.1 floor caps multi-scatter amplification at 10× L₂ — enough to fill
    // shadow sides softly without turning the whole sky into a white haze.
    let f_ms = ms_shared[0] * FRAC_4_PI;
    let l_2 = l2_shared[0] * FRAC_4_PI;
    let psi_ms = l_2 / max(vec3(1.0) - f_ms, vec3(0.1));
    textureStore(multiscatter_out, vec2<i32>(id.xy), vec4(psi_ms, 1.0));
}
