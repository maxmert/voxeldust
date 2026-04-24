// Sky-view LUT compute shader. Output: 192×108 RGBA16F.
// Each texel stores the inscattered luminance for a view ray from the observer
// (at current altitude) along a given (zenith, azimuth) direction. Azimuth is
// measured in atmosphere space relative to a world-fixed reference so the
// terminator stays stable as the camera tilts.
//
// This is rebuilt EVERY FRAME — depends on observer altitude and sun direction.
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
var skyview_out: texture_storage_2d<rgba16float, write>;

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

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(u32(SKY_VIEW_LUT_W), u32(SKY_VIEW_LUT_H));
    if id.x >= size.x || id.y >= size.y { return; }

    let uv = (vec2<f32>(id.xy) + vec2(0.5)) / vec2<f32>(size);
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;

    // Observer: place at (0, r, 0) in atmosphere space. Observer world-space
    // position is relative to planet center (km); its length is the radius.
    let observer_ws = atmo.observer_pos.xyz;
    // Clamp observer r to the atmosphere band. Above the outer shell the Bruneton
    // transmittance parameterization goes out of range, and both scattering
    // integrals are uninteresting (empty space). Sampling at outer_r - ε gives a
    // valid "sitting at the top of the atmosphere" reference that matches what
    // an observer in space should see when looking toward the limb.
    let r = clamp(length(observer_ws), inner_r + 0.001, outer_r - 0.001);

    // Decode ray direction in atmosphere space from UV.
    let za = sky_view_lut_uv_to_zenith_azimuth(inner_r, r, uv);
    let ray_as = zenith_azimuth_to_ray_dir(za.x, za.y);

    // Convert atmosphere-space ray direction into world space to dot with sun.
    let ray_ws = (atmo.world_from_atmo * vec4(ray_as, 0.0)).xyz;

    // mu = cos(zenith angle) in atmosphere space, equals ray_as.y by construction.
    let mu = ray_as.y;
    let t_max = max_atmosphere_distance(inner_r, outer_r, r, mu);

    // Ray march the sky view.
    var inscatter = vec3(0.0);
    var transmittance = vec3(1.0);
    var prev_t = 0.0;
    let n = SKY_VIEW_SAMPLES;
    let sun_dir_ws = atmo.sun_dir.xyz;
    let sun_illum = atmo.sun_color.xyz * max(atmo.quality.y, 0.0);
    let g = atmo.mie.w;
    let cos_theta = dot(ray_ws, sun_dir_ws);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, g);

    // Sample positions along the ray are expressed in atmosphere-space radial
    // coordinates (r, mu). For sun transmittance we need mu_sun at each sample;
    // we compute that from the world-space sun direction and a world-space
    // approximation of the sample's local up. Because atmosphere-space local
    // up at sample (r_i, mu_i, t_i) maps to world via world_from_atmo, and
    // the world up at the sample is simply normalize(sample_ws).
    for (var i = 0u; i < n; i++) {
        let t = t_max * (f32(i) + MIDPOINT_RATIO) / f32(n);
        let dt = t - prev_t;
        prev_t = t;
        let r_i = step_r(r, mu, t);
        let alt = r_i - inner_r;
        let m = sample_medium(atmo, alt);
        let seg_transmittance = exp(-m.extinction * dt);

        // Sample position in world space.
        let sample_ws = observer_ws + ray_ws * t;
        let up_ws = normalize(sample_ws);
        let mu_sun = dot(up_ws, sun_dir_ws);
        let hits_ground = ray_hits_ground(inner_r, r_i, mu_sun);
        var trans_to_sun = vec3(0.0);
        if !hits_ground {
            trans_to_sun = sample_transmittance(r_i, mu_sun);
        }

        let scattered = m.rayleigh_scattering * phase_r + vec3(m.mie_scattering) * phase_m;
        // Multi-scatter: add isotropic ambient lighting from ψ_ms LUT, scaled
        // by the local scattering coefficient (phaseless). This is the key
        // physics that makes the shadow side / zenith look naturally lit
        // instead of flat black.
        let scattering_iso = m.rayleigh_scattering + vec3(m.mie_scattering);
        let psi_ms = sample_multiscatter(r_i, mu_sun);
        let inscat_local = sun_illum * (trans_to_sun * scattered + psi_ms * scattering_iso);
        let s_int = (inscat_local - inscat_local * seg_transmittance) / max(m.extinction, MIN_EXTINCTION);
        inscatter += transmittance * s_int;
        transmittance *= seg_transmittance;

        if all(transmittance < vec3(0.001)) { break; }
    }

    // Add ground contribution (Lambert reflectance) if the ray hits the planet.
    if ray_hits_ground(inner_r, r, mu) {
        let t_ground = distance_to_ground(inner_r, r, mu);
        let r_g = inner_r;
        let sample_ws = observer_ws + ray_ws * t_ground;
        let up_ws = normalize(sample_ws);
        let mu_sun = dot(up_ws, sun_dir_ws);
        if mu_sun > 0.0 {
            let trans_to_sun = sample_transmittance(r_g, mu_sun);
            let trans_to_ground = sample_transmittance(r, mu) / max(sample_transmittance(r_g, mu), vec3(1e-6));
            let albedo = vec3(atmo.ozone_extra.y);
            let ground_lum = trans_to_ground * albedo * (mu_sun / PI) * sun_illum * trans_to_sun;
            inscatter += ground_lum;
        }
    }

    // Night-side airglow: faint thermospheric emission (O I 557.7 nm green line
    // on Earth; we use a muted green-cyan tint tinged by the planet's Rayleigh
    // color to keep it coherent with the daytime sky palette). Visible only on
    // the night side (sun below horizon), strongest near the horizon where the
    // line of sight crosses more atmosphere. Strength scales with atmospheric
    // density so airless bodies show no airglow.
    let observer_up = normalize(observer_ws);
    let sun_alt = dot(observer_up, sun_dir_ws);
    let night_factor = smoothstep(0.05, -0.15, sun_alt);
    if night_factor > 0.0 {
        // Above-horizon test: mu < 0 would be looking down through planet.
        let above_horizon = smoothstep(-0.2, 0.1, mu);
        // Modulate by horizon-grazing strength (more atmosphere = brighter).
        let horizon_bias = 1.0 - abs(mu);
        let rayleigh_tint = normalize(atmo.rayleigh.xyz + vec3(1e-6));
        // Weighted palette: neutral green-cyan for O I airglow + planet-tinted
        // hydrogen/oxygen recombination background.
        let airglow_color = mix(vec3(0.30, 0.85, 0.55), rayleigh_tint, 0.35);
        let airglow = airglow_color * 0.0004 * night_factor * above_horizon * (0.4 + horizon_bias);
        inscatter += airglow;
    }

    textureStore(skyview_out, vec2<i32>(id.xy), vec4(inscatter, 1.0));
}
