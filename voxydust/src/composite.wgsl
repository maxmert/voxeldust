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

    // Adaptive sample count: more samples for longer rays.
    let min_spp = atmo.quality.x;
    let max_spp = atmo.quality.y;
    let ray_length = t_end - t_start;
    let sample_count_f = clamp(
        mix(min_spp, max_spp, saturate(ray_length * 0.01)),
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

    // Atmosphere compositing (only for exterior pixels, only when enabled).
    if atmo.radii.z > 0.5 && is_exterior > 0.5 {
        let is_sky = depth < 1e-6; // reverse-Z: 0 = infinity (sky)

        // Reconstruct world-space view ray direction from depth.
        let ndc = vec4(
            in.uv.x * 2.0 - 1.0,
            (1.0 - in.uv.y) * 2.0 - 1.0,
            select(depth, 0.01, is_sky), // use small depth for sky to get valid direction
            1.0
        );
        let world_h = atmo.inv_vp * ndc;
        let world_pos = world_h.xyz / world_h.w;
        let ray_dir = normalize(world_pos);

        // Distance from camera to fragment in km (meters → km conversion).
        // For sky pixels, use a very large distance to ray-march the full atmosphere.
        let frag_dist_km = select(length(world_pos) * 0.001, 1e6, is_sky);

        // Observer position in km (from AtmosphereUniforms).
        let observer = atmo.observer_pos.xyz;

        // Ray march atmosphere.
        let sun_dir = atmo.sun_dir.xyz;
        let sun_illum = atmo.sun_color.xyz;
        let atmo_result = atmosphere_ray_march(observer, ray_dir, frag_dist_km, sun_dir, sun_illum);

        // Composite: attenuate scene by transmittance, add in-scattered light.
        color = color * atmo_result.transmittance + atmo_result.inscatter;
    }

    // Tonemapping: ACES filmic.
    color = aces_tonemap(color);

    return vec4(color, 1.0);
}
