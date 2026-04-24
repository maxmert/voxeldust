// Shared atmosphere math. Concatenated into each compute shader and the composite
// fragment shader via include_str!("atmo_common.wgsl"). WGSL has no include
// directive — this file defines types/functions to be textually prepended.
//
// Units: kilometers for radii, scale heights, distances. 1/km for scattering
// coefficients. Altitude = distance from planet center - inner_radius.
//
// Physics: Hillaire 2020 "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique" (EGSR). LUT UV parameterizations adapted from
// bevy_pbr::atmosphere (MIT/Apache) which in turn follows Bruneton 2008
// "Precomputed Atmospheric Scattering" (see bruneton_functions.wgsl in Bevy
// for the original BSD-style license that covers the parameterization).

const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679;
const PI_2: f32 = 6.28318530717958;
const FRAC_2_PI: f32 = 0.15915494309;
const MIN_EXTINCTION: vec3<f32> = vec3(1e-12);
// Sample bias within each ray march segment — 0.3 biases toward the start,
// which better matches the exponential falloff of atmospheric density.
const MIDPOINT_RATIO: f32 = 0.3;

// Per-planet atmosphere parameters + per-frame view data. Populated CPU-side
// in build_atmosphere_uniforms().
struct AtmosphereUniforms {
    // xyz = observer position relative to planet center (km). w = unused.
    observer_pos: vec4<f32>,
    // x = inner_radius (km, planet surface), y = outer_radius (km, atmo top),
    // z = enabled flag (>0.5 means atmosphere exists), w = sun angular diameter (rad).
    radii: vec4<f32>,
    // xyz = Rayleigh scattering at sea level (1/km). w = exponential scale (-1/H_rayleigh, 1/km).
    rayleigh: vec4<f32>,
    // x = Mie scattering (1/km), y = Mie absorption (1/km),
    // z = Mie density exp scale (-1/H_mie), w = Mie asymmetry g.
    mie: vec4<f32>,
    // xyz = Ozone absorption (1/km). w = layer center altitude (km).
    ozone: vec4<f32>,
    // x = ozone width (km), y = ground albedo (scalar),
    // z = multi-scatter factor (placeholder until MS LUT ships), w = unused.
    ozone_extra: vec4<f32>,
    // xyz = direction TO sun (world, normalized). w = sun disk radius (rad).
    sun_dir: vec4<f32>,
    // xyz = sun illuminance (linear RGB, exposed). w = unused.
    sun_color: vec4<f32>,
    // Inverse view-projection for world-pos reconstruction from depth.
    inv_vp: mat4x4<f32>,
    // x = width, y = height, z = 1/w, w = 1/h.
    screen: vec4<f32>,
    // x = weather Mie multiplier (clouds thickening air),
    // y = weather sun occlusion (overcast darkening),
    // z = aerial-view LUT max distance (km),
    // w = unused.
    quality: vec4<f32>,
    // Rotation that maps world direction → atmosphere space (local up = +Y).
    // Translation is zero; observer in atmosphere space is (0, r, 0).
    atmo_from_world: mat4x4<f32>,
    // Inverse of the above. Used to convert LUT-space directions back to world.
    world_from_atmo: mat4x4<f32>,
}

// LUT sizes — compile-time constants so UV math is stable across all shaders.
// Values match Hillaire's recommendations and Bevy's defaults.
const TRANSMITTANCE_LUT_W: f32 = 256.0;
const TRANSMITTANCE_LUT_H: f32 = 64.0;
const SKY_VIEW_LUT_W: f32 = 192.0;
const SKY_VIEW_LUT_H: f32 = 108.0;
const AERIAL_LUT_W: f32 = 32.0;
const AERIAL_LUT_H: f32 = 32.0;
const AERIAL_LUT_D: f32 = 32.0;

// Sample counts — reasonable defaults; could be exposed as a graphics setting later.
const TRANSMITTANCE_SAMPLES: u32 = 40u;
const SKY_VIEW_SAMPLES: u32 = 30u;
const AERIAL_SAMPLES_PER_SLICE: u32 = 2u;

// ---------------------------------------------------------------------------
// Medium density sampling — Rayleigh + Mie + Ozone.
// Same math as the original composite.wgsl; kept analytical (no density LUT
// needed since exp() is trivially cheap and our params are already seeded).
// ---------------------------------------------------------------------------

struct MediumSample {
    // Per-channel extinction (1/km) — Rayleigh + Mie + Ozone.
    extinction: vec3<f32>,
    // Per-channel scattering (1/km) — Rayleigh + Mie. Ozone is absorbing-only.
    rayleigh_scattering: vec3<f32>,
    mie_scattering: f32,
}

fn sample_medium(uniforms: AtmosphereUniforms, altitude_km: f32) -> MediumSample {
    // Rayleigh: exponential decay with altitude.
    let rayleigh_density = exp(uniforms.rayleigh.w * altitude_km);
    // Mie: exponential decay, modulated by weather (fog/clear).
    let mie_density = exp(uniforms.mie.z * altitude_km) * max(uniforms.quality.x, 1.0);
    // Ozone: piecewise-linear tent around ozone center altitude.
    let oz_center = uniforms.ozone.w;
    let oz_width = uniforms.ozone_extra.x;
    var oz_density = 0.0;
    if oz_width > 0.0 {
        oz_density = max(0.0, 1.0 - abs(altitude_km - oz_center) / (oz_width * 0.5));
    }

    let rayleigh_sc = rayleigh_density * uniforms.rayleigh.xyz;
    let mie_sc = mie_density * uniforms.mie.x;
    let mie_ext = mie_density * (uniforms.mie.x + uniforms.mie.y);
    let oz_ext = oz_density * uniforms.ozone.xyz;

    var s: MediumSample;
    s.rayleigh_scattering = rayleigh_sc;
    s.mie_scattering = mie_sc;
    s.extinction = rayleigh_sc + vec3(mie_ext) + oz_ext;
    return s;
}

// ---------------------------------------------------------------------------
// Phase functions.
// ---------------------------------------------------------------------------

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

// Cornette-Shanks phase (improved Henyey-Greenstein, energy-conserving).
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
}

// ---------------------------------------------------------------------------
// Ray-sphere primitives.
// ---------------------------------------------------------------------------

// Returns (t_near, t_far). May be negative if behind origin.
fn ray_sphere_both(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 { return vec2(-1.0, -1.0); }
    let sd = sqrt(disc);
    return vec2(-b - sd, -b + sd);
}

// ---------------------------------------------------------------------------
// Bruneton UV parameterizations — ported from bevy_pbr::atmosphere::bruneton_functions
// (BSD-style license, Eric Bruneton / INRIA — see header comment in that file).
// ---------------------------------------------------------------------------

fn distance_to_top_atmosphere(inner_r: f32, outer_r: f32, r: f32, mu: f32) -> f32 {
    let d = max(r * r * (mu * mu - 1.0) + outer_r * outer_r, 0.0);
    return max(-r * mu + sqrt(d), 0.0);
}

fn distance_to_ground(inner_r: f32, r: f32, mu: f32) -> f32 {
    let d = max(r * r * (mu * mu - 1.0) + inner_r * inner_r, 0.0);
    return max(-r * mu - sqrt(d), 0.0);
}

fn ray_hits_ground(inner_r: f32, r: f32, mu: f32) -> bool {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + inner_r * inner_r >= 0.0;
}

fn max_atmosphere_distance(inner_r: f32, outer_r: f32, r: f32, mu: f32) -> f32 {
    let t_top = distance_to_top_atmosphere(inner_r, outer_r, r, mu);
    let t_bot = distance_to_ground(inner_r, r, mu);
    return select(t_top, t_bot, ray_hits_ground(inner_r, r, mu));
}

// Transmittance LUT: u = angular parameter, v = altitude.
fn transmittance_lut_r_mu_to_uv(inner_r: f32, outer_r: f32, r: f32, mu: f32) -> vec2<f32> {
    let H = sqrt(outer_r * outer_r - inner_r * inner_r);
    let rho = sqrt(max(r * r - inner_r * inner_r, 0.0));
    let d = distance_to_top_atmosphere(inner_r, outer_r, r, mu);
    let d_min = outer_r - r;
    let d_max = rho + H;
    let u = (d - d_min) / max(d_max - d_min, 1e-6);
    let v = rho / max(H, 1e-6);
    return vec2(u, v);
}

fn transmittance_lut_uv_to_r_mu(inner_r: f32, outer_r: f32, uv: vec2<f32>) -> vec2<f32> {
    let H = sqrt(outer_r * outer_r - inner_r * inner_r);
    let rho = H * uv.y;
    let r = sqrt(rho * rho + inner_r * inner_r);
    let d_min = outer_r - r;
    let d_max = rho + H;
    let d = d_min + uv.x * (d_max - d_min);
    var mu: f32;
    if d == 0.0 {
        mu = 1.0;
    } else {
        mu = (H * H - rho * rho - d * d) / (2.0 * r * max(d, 1e-6));
    }
    return vec2(r, clamp(mu, -1.0, 1.0));
}

// Sub-UV bias: shifts integer texel centers so the LUT does not alias edges.
fn unit_to_sub_uvs(val: vec2<f32>, res: vec2<f32>) -> vec2<f32> {
    return (val + 0.5 / res) * (res / (res + 1.0));
}

fn sub_uvs_to_unit(val: vec2<f32>, res: vec2<f32>) -> vec2<f32> {
    return (val - 0.5 / res) * (res / (res - 1.0));
}

// Sky-view LUT: non-linear horizon-concentrated parameterization.
// Reference: Hillaire 2020 §5.3 + Bevy's implementation (MIT/Apache).
// `azimuth` is measured in atmosphere space relative to the world-fixed +X axis.
fn sky_view_lut_uv_to_zenith_azimuth(inner_r: f32, r: f32, uv: vec2<f32>) -> vec2<f32> {
    let res = vec2(SKY_VIEW_LUT_W, SKY_VIEW_LUT_H);
    let adj = sub_uvs_to_unit(vec2(uv.x, 1.0 - uv.y), res);
    let azimuth = (adj.x - 0.5) * PI_2;
    let v_horizon = sqrt(max(r * r - inner_r * inner_r, 0.0));
    let cos_beta = v_horizon / r;
    let beta = acos(clamp(cos_beta, -1.0, 1.0));
    let horizon_zenith = PI - beta;
    let t = abs(2.0 * (adj.y - 0.5));
    let l = sign(adj.y - 0.5) * HALF_PI * t * t;
    return vec2(horizon_zenith - l, azimuth);
}

fn sky_view_lut_r_mu_azimuth_to_uv(inner_r: f32, r: f32, mu: f32, azimuth: f32) -> vec2<f32> {
    let res = vec2(SKY_VIEW_LUT_W, SKY_VIEW_LUT_H);
    let u = azimuth * FRAC_2_PI + 0.5;
    let v_horizon = sqrt(max(r * r - inner_r * inner_r, 0.0));
    let cos_beta = v_horizon / r;
    let beta = acos(clamp(cos_beta, -1.0, 1.0));
    let horizon_zenith = PI - beta;
    let view_zenith = acos(clamp(mu, -1.0, 1.0));
    let l = view_zenith - horizon_zenith;
    let abs_l = abs(l);
    let v = 0.5 + 0.5 * sign(l) * sqrt(abs_l / HALF_PI);
    return unit_to_sub_uvs(vec2(u, v), res);
}

// Multi-scatter LUT parameterization. Matches atmo_multiscatter.wgsl.
fn multiscatter_r_mu_to_uv(inner_r: f32, outer_r: f32, r: f32, mu: f32) -> vec2<f32> {
    let u = 0.5 + 0.5 * mu;
    let v = saturate((r - inner_r) / max(outer_r - inner_r, 1e-6));
    return unit_to_sub_uvs(vec2(u, v), vec2(32.0, 32.0));
}

// Spherical direction in atmosphere space (local up = +Y, azimuth around Y axis).
fn zenith_azimuth_to_ray_dir(zenith: f32, azimuth: f32) -> vec3<f32> {
    let sz = sin(zenith);
    let cz = cos(zenith);
    let sa = sin(azimuth);
    let ca = cos(azimuth);
    return vec3(sa * sz, cz, -ca * sz);
}

// Given position on ray (r, mu, t), return new radius after traveling t along ray.
fn step_r(r: f32, mu: f32, t: f32) -> f32 {
    return sqrt(t * t + 2.0 * r * mu * t + r * r);
}
