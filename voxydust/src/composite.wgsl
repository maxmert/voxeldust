// HDR composite pass: atmospheric scattering (Hillaire 2020) + tonemapping.
//
// Atmosphere ray march adapted from webgpu-sky-atmosphere (proven WGSL port of
// Hillaire/UE5 sky atmosphere). All scattering parameters are per-planet,
// derived deterministically from the planet seed. No hardcoded constants.
//
// References:
// - Hillaire 2020: "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
// - webgpu-sky-atmosphere: github.com/JolifantoBambla/webgpu-sky-atmosphere
// - Bevy: crates/bevy_pbr/src/atmosphere/

const PI: f32 = 3.14159265358979;

struct CompositeParams {
    screen_dims: vec4<f32>,  // xy = width, height; zw = 1/width, 1/height
}

struct AtmosphereUniforms {
    observer_pos: vec4<f32>,          // xyz = observer relative to planet center (km), w = unused
    radii: vec4<f32>,                 // x = planet radius (km), y = atmo top radius (km), z = enabled, w = unused
    rayleigh: vec4<f32>,              // xyz = scattering coeffs (1/km), w = density exp scale (-1/H_km)
    mie: vec4<f32>,                   // x = scatter, y = absorb, z = density exp scale, w = g (anisotropy)
    ozone: vec4<f32>,                 // xyz = absorption coeffs (1/km), w = center alt (km)
    ozone_extra: vec4<f32>,           // x = width (km), y = ground albedo, z = multi_scatter_factor, w = unused
    sun_dir: vec4<f32>,               // xyz = direction TO sun (normalized), w = sun disk angular diameter
    sun_color: vec4<f32>,             // xyz = sun illuminance (linear RGB), w = unused
    inv_vp: mat4x4<f32>,             // inverse view-projection for depth reconstruction
    screen: vec4<f32>,                // x = width, y = height, z = 1/w, w = 1/h
    quality: vec4<f32>,               // x = min samples, y = max samples, z = weather_mie_mult, w = weather_sun_occlusion
}

@group(0) @binding(0)
var hdr_scene: texture_2d<f32>;

@group(0) @binding(1)
var screen_sampler: sampler;

@group(0) @binding(2)
var<uniform> params: CompositeParams;

@group(0) @binding(3)
var depth_tex: texture_depth_2d;

@group(0) @binding(4)
var<uniform> atmo: AtmosphereUniforms;

// Cloud noise textures + uniform.
@group(0) @binding(5)
var cloud_shape_tex: texture_3d<f32>;

@group(0) @binding(6)
var cloud_detail_tex: texture_3d<f32>;

@group(0) @binding(7)
var cloud_sampler: sampler;

struct CloudUniforms {
    observer_pos: vec4<f32>,     // xyz = observer (km), w = game_time
    geometry: vec4<f32>,         // x=planet_r, y=cloud_base, z=cloud_thick, w=enabled
    density_params: vec4<f32>,   // x=coverage, y=density_scale, z=cloud_type, w=absorption
    wind: vec4<f32>,             // xyz=wind_vel (km/s), w=wind_shear
    scatter: vec4<f32>,          // xyz=scatter_color, w=weather_scale (km)
    sun: vec4<f32>,              // xyz=sun_dir, w=sun_intensity
    sun_color: vec4<f32>,        // xyz=sun_color_rgb, w=base_noise_freq
    noise_params: vec4<f32>,     // x=shape_scale, y=detail_scale, z=weather_octaves, w=unused
}

@group(0) @binding(8)
var<uniform> clouds: CloudUniforms;

@group(0) @binding(9)
var weather_map: texture_2d<f32>;

// ---------------------------------------------------------------------------
// Fullscreen vertex shader (vertexless triangle)
// ---------------------------------------------------------------------------

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FullscreenOutput {
    var out: FullscreenOutput;
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.position = vec4(x, y, 0.0, 1.0);
    out.uv = vec2(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ---------------------------------------------------------------------------
// Ray-sphere intersection (from webgpu-sky-atmosphere/intersection.wgsl)
// ---------------------------------------------------------------------------

/// Returns distance to closest intersection of ray (o,d) with sphere at origin with radius r.
/// Returns -1.0 if no intersection or both behind ray origin.
fn ray_sphere_nearest(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
    let a = dot(d, d);
    let b = 2.0 * dot(d, o);
    let c = dot(o, o) - r * r;
    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 || a == 0.0 { return -1.0; }
    let sd = sqrt(delta);
    let t0 = (-b - sd) / (2.0 * a);
    let t1 = (-b + sd) / (2.0 * a);
    if t0 < 0.0 && t1 < 0.0 { return -1.0; }
    if t0 < 0.0 { return max(0.0, t1); }
    if t1 < 0.0 { return max(0.0, t0); }
    return max(0.0, min(t0, t1));
}

/// Returns (t_near, t_far) of ray-sphere intersection. Both may be negative.
fn ray_sphere(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let a = dot(d, d);
    let b = 2.0 * dot(d, o);
    let c = dot(o, o) - r * r;
    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 { return vec2(-1.0, -1.0); }
    let sd = sqrt(delta);
    return vec2((-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a));
}

/// Check if ray hits planet (returns true if the planet body blocks the ray).
fn ray_hits_planet(o: vec3<f32>, d: vec3<f32>, planet_r: f32) -> bool {
    let b = dot(o, d);
    let c = dot(o, o) - planet_r * planet_r;
    let delta = b * b - c;
    return delta >= 0.0 && -b - sqrt(delta) > 0.0;
}

// ---------------------------------------------------------------------------
// Medium density sampling (from webgpu-sky-atmosphere/medium.wgsl)
// All parameters from AtmosphereUniforms — nothing hardcoded.
// ---------------------------------------------------------------------------

struct MediumSample {
    scattering: vec3<f32>,
    extinction: vec3<f32>,
    mie_scattering: f32,
    rayleigh_scattering: vec3<f32>,
}

fn sample_medium(height_km: f32) -> MediumSample {
    // Rayleigh density: exponential decay with altitude.
    let rayleigh_density = exp(atmo.rayleigh.w * height_km); // w = -1/H_rayleigh

    // Mie density: exponential decay with altitude, modulated by weather.
    let mie_density = exp(atmo.mie.z * height_km) * atmo.quality.z; // z = -1/H_mie, quality.z = weather_mie_mult

    // Ozone density: piecewise linear tent around center altitude.
    let ozone_center = atmo.ozone.w; // center altitude (km)
    let ozone_width = atmo.ozone_extra.x; // width (km)
    var ozone_density = 0.0;
    if ozone_width > 0.0 {
        ozone_density = max(0.0, 1.0 - abs(height_km - ozone_center) / (ozone_width * 0.5));
    }

    var s: MediumSample;
    s.rayleigh_scattering = rayleigh_density * atmo.rayleigh.xyz;
    s.mie_scattering = mie_density * atmo.mie.x;

    s.scattering = s.rayleigh_scattering + vec3(s.mie_scattering);

    // Extinction = scattering + absorption.
    let mie_extinction = mie_density * (atmo.mie.x + atmo.mie.y); // scatter + absorb
    let ozone_extinction = ozone_density * atmo.ozone.xyz;
    s.extinction = s.rayleigh_scattering + vec3(mie_extinction) + ozone_extinction;

    return s;
}

// ---------------------------------------------------------------------------
// Phase functions (from webgpu-sky-atmosphere/phase.wgsl)
// ---------------------------------------------------------------------------

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

/// Cornette-Shanks phase function (improved Henyey-Greenstein, energy-conserving).
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
}

// ---------------------------------------------------------------------------
// Atmosphere ray march (adapted from webgpu-sky-atmosphere integrate_scattered_luminance)
// Single-scattering with analytical per-segment integration.
// ---------------------------------------------------------------------------

struct AtmosphereResult {
    inscatter: vec3<f32>,
    transmittance: vec3<f32>,
}

fn atmosphere_ray_march(
    ray_origin: vec3<f32>,   // observer position relative to planet center (km)
    ray_dir: vec3<f32>,      // normalized view direction
    max_dist: f32,           // distance to geometry (km), or very large for sky
    sun_dir: vec3<f32>,      // direction toward sun (normalized)
    sun_illuminance: vec3<f32>,
) -> AtmosphereResult {
    var result: AtmosphereResult;
    result.inscatter = vec3(0.0);
    result.transmittance = vec3(1.0);

    let planet_r = atmo.radii.x;
    let atmo_r = atmo.radii.y;

    // Find intersection with atmosphere shell.
    let t_atmo = ray_sphere(ray_origin, ray_dir, atmo_r);
    if t_atmo.x > t_atmo.y { return result; } // no intersection

    var t_start = max(t_atmo.x, 0.0);
    var t_end = t_atmo.y;

    // Clamp to geometry depth.
    if max_dist < t_end { t_end = max_dist; }
    if t_start >= t_end { return result; }

    // Check planet intersection to clamp ray.
    let t_planet = ray_sphere_nearest(ray_origin, ray_dir, planet_r);
    if t_planet > 0.0 && t_planet < t_end { t_end = t_planet; }

    // Adaptive sample count: more samples for longer rays, fewer for nearby terrain.
    let min_spp = atmo.quality.x;
    let max_spp = atmo.quality.y;
    let ray_length = t_end - t_start;

    // Skip atmosphere for very close fragments (< 0.1 km) — no visible scattering.
    if ray_length < 0.1 { return result; }

    // Tighter scaling: reach max samples only for long rays (> 200km).
    let sample_count_f = clamp(
        mix(min_spp, max_spp, saturate(ray_length * 0.005)),
        min_spp, max_spp
    );
    let sample_count = u32(sample_count_f);
    if sample_count == 0u { return result; }

    let dt = ray_length / sample_count_f;

    // Pre-compute phase values (constant along the ray).
    let cos_theta = dot(ray_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, atmo.mie.w);

    // Secondary ray march sample count for sun transmittance.
    let light_samples = 4u;

    for (var i = 0u; i < sample_count; i++) {
        let t = t_start + (f32(i) + 0.5) * dt;
        let sample_pos = ray_origin + ray_dir * t;
        let sample_height = length(sample_pos);
        let altitude_km = sample_height - planet_r;

        let medium = sample_medium(altitude_km);
        let sample_transmittance = exp(-medium.extinction * dt);

        // Secondary ray: transmittance from sample point to sun (through atmosphere).
        var transmittance_to_sun = vec3(1.0);
        if !ray_hits_planet(sample_pos, sun_dir, planet_r) {
            let t_sun = ray_sphere(sample_pos, sun_dir, atmo_r);
            if t_sun.y > 0.0 {
                let sun_dt = t_sun.y / f32(light_samples);
                var sun_optical_depth = vec3(0.0);
                for (var j = 0u; j < light_samples; j++) {
                    let sun_t = (f32(j) + 0.5) * sun_dt;
                    let sun_pos = sample_pos + sun_dir * sun_t;
                    let sun_alt = length(sun_pos) - planet_r;
                    if sun_alt < 0.0 { break; }
                    let sun_medium = sample_medium(sun_alt);
                    sun_optical_depth += sun_medium.extinction * sun_dt;
                }
                transmittance_to_sun = exp(-sun_optical_depth);
            }
        } else {
            transmittance_to_sun = vec3(0.0); // planet blocks sun
        }

        // Phase-weighted scattering coefficients.
        let scattered = medium.rayleigh_scattering * phase_r
                       + vec3(medium.mie_scattering) * phase_m;

        // Weather sun occlusion (future: cloud cover).
        let sun_occlusion = atmo.quality.w;

        let in_scattered_luminance = sun_illuminance * sun_occlusion * transmittance_to_sun * scattered;

        // Analytical integration of scattering over the segment (constant-within-segment).
        // Integral of S * exp(-extinction * t) dt from 0 to dt = (S - S*T_segment) / extinction.
        let scattering_integral = (in_scattered_luminance - in_scattered_luminance * sample_transmittance)
            / max(medium.extinction, vec3(1e-10));

        result.inscatter += result.transmittance * scattering_integral;
        result.transmittance *= sample_transmittance;

        // Early termination when transmittance is negligible.
        if all(result.transmittance < vec3(0.001)) { break; }
    }

    return result;
}

// ---------------------------------------------------------------------------
// ACES filmic tonemapping
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Volumetric Cloud Ray March (Nubis/HZD technique)
//
// Density function: Perlin-Worley base shape + Worley detail erosion
// Lighting: Beer-Lambert + multi-scattering octave approximation
// All parameters from CloudUniforms (per-planet, seed-derived)
// ---------------------------------------------------------------------------

fn remap_cloud(value: f32, old_lo: f32, old_hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + saturate((value - old_lo) / (old_hi - old_lo)) * (new_hi - new_lo);
}

/// Height gradient for cloud type blending (Nubis).
/// type_blend: 0=stratus(flat), 0.5=cumulus(puffy), 1=cumulonimbus(towering).
fn cloud_height_gradient(height_frac: f32, type_blend: f32) -> f32 {
    // Stratus: thin layer concentrated at base.
    let stratus = saturate(remap_cloud(height_frac, 0.0, 0.1, 0.0, 1.0))
                * saturate(remap_cloud(height_frac, 0.2, 0.3, 1.0, 0.0));
    // Cumulus: puffy, peaks in middle.
    let cumulus = saturate(remap_cloud(height_frac, 0.0, 0.15, 0.0, 1.0))
               * saturate(remap_cloud(height_frac, 0.5, 0.9, 1.0, 0.0));
    // Cumulonimbus: towering, fills most of the layer.
    let cb = saturate(remap_cloud(height_frac, 0.0, 0.1, 0.0, 1.0))
           * saturate(remap_cloud(height_frac, 0.7, 1.0, 1.0, 0.0));

    // Blend between types.
    if type_blend < 0.5 {
        return mix(stratus, cumulus, type_blend * 2.0);
    }
    return mix(cumulus, cb, (type_blend - 0.5) * 2.0);
}

/// Sample the precomputed weather map for cloud coverage and type at a world position.
/// The weather map is generated on CPU (Hadley cells + pressure cells) and uploaded as
/// a 2D texture in equirectangular projection. All weather complexity lives on the CPU —
/// the shader just reads the result.
///
/// Returns: R=coverage, G=cloud_type, B=precipitation, A=wind_modifier
fn sample_weather(pos: vec3<f32>) -> vec4<f32> {
    let dir = normalize(pos);
    let lon = atan2(dir.z, dir.x);                // [-PI, PI]
    let lat = asin(clamp(dir.y, -1.0, 1.0));      // [-PI/2, PI/2]
    let uv = vec2(
        lon / (2.0 * PI) + 0.5,                   // [0, 1]
        lat / PI + 0.5                             // [0, 1]
    );
    return textureSample(weather_map, cloud_sampler, uv);
}

fn weather_coverage(pos: vec3<f32>, planet_r: f32) -> f32 {
    return sample_weather(pos).r;
}

/// Sample cloud density at a world position (km, relative to planet center).
fn cloud_density(pos: vec3<f32>) -> f32 {
    let planet_r = clouds.geometry.x;
    let cloud_base = clouds.geometry.y;
    let cloud_thick = clouds.geometry.z;

    let altitude = length(pos) - planet_r;
    let height_frac = (altitude - cloud_base) / cloud_thick;
    if height_frac <= 0.0 || height_frac >= 1.0 { return 0.0; }

    // Single weather map sample: coverage (R) + cloud_type (G).
    // Avoids redundant texture lookup (was sampled twice before).
    let weather = sample_weather(pos);
    let coverage = weather.r;
    if coverage < 0.01 { return 0.0; }

    let cloud_type = weather.g;
    let gradient = cloud_height_gradient(height_frac, cloud_type);
    if gradient < 0.001 { return 0.0; }

    // Wind offset for cloud scrolling.
    let time = clouds.observer_pos.w;
    let wind_offset = clouds.wind.xyz * time;

    // Sample 3D shape noise (tileable, scrolled by wind).
    let shape_scale = clouds.noise_params.x;
    let shape_uv = (pos + wind_offset) * shape_scale;
    let shape = textureSample(cloud_shape_tex, cloud_sampler, shape_uv);

    // Combine shape octaves: Perlin-Worley base + Worley fBm erosion.
    let worley_fbm = shape.g * 0.625 + shape.b * 0.25 + shape.a * 0.125;
    let base_shape = remap_cloud(shape.r, worley_fbm - 1.0, 1.0, 0.0, 1.0);

    // Early exit: skip expensive detail sampling if base shape is negligible.
    if base_shape < 0.05 { return 0.0; }

    // Apply height gradient and coverage.
    var density = base_shape * gradient;
    density = remap_cloud(density, 1.0 - coverage, 1.0, 0.0, 1.0) * coverage;
    if density <= 0.0 { return 0.0; }

    // Detail erosion (fine edge breakup).
    let detail_scale = clouds.noise_params.y;
    let sheared_wind = wind_offset * clouds.wind.w; // wind shear for upper cloud motion
    let detail_uv = (pos + sheared_wind) * detail_scale;
    let detail = textureSample(cloud_detail_tex, cloud_sampler, detail_uv);
    let detail_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;

    // Erode more at cloud tops, less at base.
    let detail_mod = mix(detail_fbm, 1.0 - detail_fbm, saturate(height_frac * 2.0));
    density = remap_cloud(density, detail_mod * 0.2, 1.0, 0.0, 1.0);

    return max(density, 0.0) * clouds.density_params.y; // density_scale
}

/// Henyey-Greenstein phase function for cloud scattering.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

struct CloudResult {
    luminance: vec3<f32>,
    transmittance: f32,
}

fn cloud_ray_march(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    max_dist: f32,
) -> CloudResult {
    var result: CloudResult;
    result.luminance = vec3(0.0);
    result.transmittance = 1.0;

    let planet_r = clouds.geometry.x;
    let cloud_base = clouds.geometry.y;
    let cloud_thick = clouds.geometry.z;
    let shell_inner = planet_r + cloud_base;
    let shell_outer = planet_r + cloud_base + cloud_thick;

    // Intersect with cloud spherical shell.
    let t_inner = ray_sphere(ray_origin, ray_dir, shell_inner);
    let t_outer = ray_sphere(ray_origin, ray_dir, shell_outer);

    // Compute ray segment through cloud layer.
    var t_start = max(max(t_inner.x, 0.0), max(t_outer.x, 0.0));
    var t_end = 0.0;
    let obs_height = length(ray_origin);

    if obs_height < shell_inner {
        // Below clouds: enter at inner sphere, exit at inner sphere (up then down).
        t_start = max(t_inner.y, 0.0); // far intersection with inner sphere
        t_end = max(t_outer.y, 0.0);   // far intersection with outer sphere... hmm
        // Actually below clouds: ray goes UP through inner, through cloud layer, hits outer.
        t_start = max(t_inner.x, 0.0);
        if t_inner.x < 0.0 { t_start = 0.0; } // inside inner sphere? shouldn't happen
        t_end = t_outer.y;
    } else if obs_height > shell_outer {
        // Above clouds (space/orbit): enter at outer sphere, exit at outer sphere.
        if t_outer.x < 0.0 { return result; } // ray misses atmosphere entirely
        t_start = t_outer.x;
        t_end = t_outer.y;
        // But clamp to inner sphere if ray passes through planet.
        if t_inner.x > 0.0 { t_end = min(t_end, t_inner.x); }
    } else {
        // Inside cloud layer.
        t_start = 0.0;
        t_end = t_outer.y;
        if t_inner.x > 0.0 { t_end = min(t_end, t_inner.x); }
    }

    t_end = min(t_end, max_dist);
    if t_start >= t_end || t_end <= 0.0 { return result; }

    // Ray march parameters.
    let ray_length = t_end - t_start;
    let max_steps = 64u;
    let dt = ray_length / f32(max_steps);

    let cos_angle = dot(ray_dir, clouds.sun.xyz);
    let phase = max(hg_phase(cos_angle, 0.8), hg_phase(cos_angle, -0.5));
    let absorption = clouds.density_params.w;
    let sun_color_val = clouds.sun_color.xyz * clouds.sun.w;
    let scatter_color = clouds.scatter.xyz;

    for (var i = 0u; i < max_steps; i++) {
        let t = t_start + (f32(i) + 0.5) * dt;
        let sample_pos = ray_origin + ray_dir * t;
        let density = cloud_density(sample_pos);

        if density > 0.001 {
            let sample_transmittance = exp(-density * absorption * dt);

            // Smooth day/night terminator based on sun angle at this cloud position.
            // Real terminators are soft (~200km transition zone from atmospheric scattering).
            // Use dot(surface_normal, sun_direction) with a smooth falloff.
            let sun_dir = clouds.sun.xyz;
            let cloud_normal = normalize(sample_pos); // radial direction = surface normal
            let sun_angle = dot(cloud_normal, sun_dir);
            // smoothstep from -0.1 to 0.15: creates a ~25° soft transition zone.
            // Negative threshold allows light to wrap slightly past the geometric terminator
            // (atmospheric light bending effect).
            let planet_shadow = smoothstep(-0.1, 0.15, sun_angle);

            // Direct sun lighting with multi-scattering approximation.
            var direct_light = vec3(0.0);
            if planet_shadow > 0.001 {
                var oct_atten = 1.0;
                var oct_contrib = 1.0;
                var oct_phase = phase;
                for (var oct = 0u; oct < 3u; oct++) {
                    let ext = density * oct_atten * absorption;
                    let beer = exp(-ext * dt * 6.0);
                    let powder = 1.0 - exp(-ext * dt * 12.0);
                    direct_light += planet_shadow * beer * mix(0.7, 1.0, powder) * oct_phase * oct_contrib * scatter_color * sun_color_val;
                    oct_atten *= 0.5;
                    oct_contrib *= 0.5;
                    oct_phase = mix(oct_phase, 1.0 / (4.0 * PI), 0.5);
                }
            }

            // Analytical integration of direct lighting through the cloud segment.
            let extinction = density * absorption;
            let direct_integrated = (direct_light - direct_light * sample_transmittance) / max(extinction, 0.001);

            // Ambient: very dim, only near the terminator (atmospheric scattering).
            // Night side gets effectively zero — no light sources means pitch black clouds.
            let ambient = scatter_color * 0.008 * planet_shadow * planet_shadow * (1.0 - sample_transmittance);

            result.luminance += result.transmittance * (direct_integrated + ambient);
            result.transmittance *= sample_transmittance;

            if result.transmittance < 0.01 { break; }
        }
    }

    return result;
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color * (color * 2.51 + vec3(0.03)) / (color * (color * 2.43 + vec3(0.59)) + vec3(0.14));
}

// ---------------------------------------------------------------------------
// Fragment shader: atmosphere composite + tonemap
// ---------------------------------------------------------------------------

@fragment
fn fs_composite(in: FullscreenOutput) -> @location(0) vec4<f32> {
    // Sample HDR scene and depth.
    let hdr = textureSample(hdr_scene, screen_sampler, in.uv);
    let hdr_color = hdr.rgb;
    let is_exterior = hdr.a; // alpha: 1.0 = exterior (apply atmosphere), 0.0 = ship interior (skip)
    let depth = textureLoad(depth_tex, vec2<i32>(in.position.xy), 0);

    var color = hdr_color;

    // Reconstruct world-space ray for atmosphere + clouds (shared computation).
    let is_sky = depth < 1e-6; // reverse-Z: 0 = infinity (sky)
    let ndc = vec4(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,
        select(depth, 0.01, is_sky),
        1.0
    );
    let world_h = atmo.inv_vp * ndc;
    let world_pos = world_h.xyz / world_h.w;
    let ray_dir = normalize(world_pos);
    let frag_dist_km = select(length(world_pos) * 0.001, 1e6, is_sky);

    // Atmosphere compositing (only for exterior pixels, only when enabled).
    if atmo.radii.z > 0.5 && is_exterior > 0.5 {
        let observer = atmo.observer_pos.xyz;
        let sun_dir = atmo.sun_dir.xyz;
        let sun_illum = atmo.sun_color.xyz;
        let atmo_result = atmosphere_ray_march(observer, ray_dir, frag_dist_km, sun_dir, sun_illum);
        color = color * atmo_result.transmittance + atmo_result.inscatter;
    }

    // Volumetric cloud compositing (Nubis technique).
    if clouds.geometry.w > 0.5 && is_exterior > 0.5 {
        let cloud_observer = clouds.observer_pos.xyz;
        let cloud_result = cloud_ray_march(cloud_observer, ray_dir, frag_dist_km);
        color = color * cloud_result.transmittance + cloud_result.luminance;
    }

    // Tonemapping: ACES filmic.
    color = aces_tonemap(color);

    return vec4(color, 1.0);
}
