// HDR composite pass: atmosphere LUT sampling + volumetric clouds + ACES tonemap.
//
// Atmosphere rendering uses four LUTs (computed per-frame in compute passes):
//   - Transmittance LUT (256×64): altitude × zenith angle → transmittance to top.
//   - Sky-view LUT (192×108): inscatter for infinity rays from current altitude.
//   - Aerial-perspective LUT (32×32×32): inscatter + transmittance for in-scene rays.
// (Multi-scatter LUT is a planned Phase 1.5 addition — currently single-scatter only.)
//
// Cloud compositing continues to ray-march on the fragment (Nubis-style) — covered
// in Phase 3 of the atmosphere plan.
//
// Shared atmosphere math is concatenated from atmo_common.wgsl at include_str! time.

// ---------------------------------------------------------------------------
// Composite-specific bindings (group 0).
// ---------------------------------------------------------------------------

struct CompositeParams {
    screen_dims: vec4<f32>,  // xy = width, height; zw = 1/width, 1/height
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

// Atmosphere LUTs (group 1).
@group(1) @binding(0)
var transmittance_lut: texture_2d<f32>;

@group(1) @binding(1)
var sky_view_lut: texture_2d<f32>;

@group(1) @binding(2)
var aerial_view_lut: texture_3d<f32>;

@group(1) @binding(3)
var lut_sampler: sampler;

// ---------------------------------------------------------------------------
// Fullscreen vertex shader (vertexless triangle).
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
// LUT sampling helpers.
// ---------------------------------------------------------------------------

fn sample_transmittance_lut_rm(r: f32, mu: f32) -> vec3<f32> {
    let inner_r = atmo.radii.x;
    let outer_r = atmo.radii.y;
    let uv = transmittance_lut_r_mu_to_uv(inner_r, outer_r, r, mu);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

// Segment transmittance using two transmittance LUT samples (Bruneton method).
// Returns transmittance of the atmosphere between a camera at (r, mu) and a
// point at distance t along the ray.
fn sample_transmittance_segment(r: f32, mu: f32, t: f32) -> vec3<f32> {
    let inner_r = atmo.radii.x;
    let r_t = step_r(r, mu, t);
    let mu_t = clamp((r * mu + t) / r_t, -1.0, 1.0);
    if ray_hits_ground(inner_r, r, mu) {
        let num = sample_transmittance_lut_rm(r_t, -mu_t);
        let den = sample_transmittance_lut_rm(r, -mu);
        return min(num / max(den, vec3(1e-6)), vec3(1.0));
    } else {
        let num = sample_transmittance_lut_rm(r, mu);
        let den = sample_transmittance_lut_rm(r_t, mu_t);
        return min(num / max(den, vec3(1e-6)), vec3(1.0));
    }
}

// Sky-view LUT sample. Takes atmosphere-space ray direction.
fn sample_skyview(r: f32, ray_dir_as: vec3<f32>) -> vec3<f32> {
    let inner_r = atmo.radii.x;
    let mu = ray_dir_as.y;
    // Azimuth in atmosphere space, measured in the XZ plane (world-fixed ref).
    let azimuth = atan2(ray_dir_as.x, -ray_dir_as.z);
    let uv = sky_view_lut_r_mu_azimuth_to_uv(inner_r, r, mu, azimuth);
    return textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb;
}

// Aerial-view LUT sample. uv is screen space; t is distance to geometry in km.
// Returns (inscatter, mean_transmittance).
fn sample_aerial(uv: vec2<f32>, t: f32) -> vec4<f32> {
    let t_max = atmo.quality.z;
    let num_slices = AERIAL_LUT_D;
    // Offset by half a slice so texel centers align with integration boundaries.
    let w = saturate(t / t_max - 0.5 / num_slices);
    let v4 = textureSampleLevel(aerial_view_lut, lut_sampler, vec3(uv, w), 0.0);
    // Fade in from zero across the first slice (before any integration).
    let t_slice = t_max / num_slices;
    let fade = saturate(t / t_slice);
    let inscatter = exp(v4.rgb) * fade;
    // Transmittance fades to 1 at the camera (zero distance).
    let transm = mix(1.0, v4.a, fade);
    return vec4(inscatter, transm);
}

// Sun disc (Allen 1973 limb darkening) + atmospheric corona/halo. The halo is
// a multi-stage exponential falloff that simulates forward-scattered sunlight
// near the disc — this is the dramatic bright "sun glow" effect visible
// around the solar disc when looking directly at it, and is the approximation
// we use in lieu of full physical bloom.
//
// Flux-preserving disc + halo scaled to roughly match cinematic reference.
fn sample_sun_disc(ray_dir_ws: vec3<f32>) -> vec3<f32> {
    let sun_dir = atmo.sun_dir.xyz;
    let sun_radius = atmo.radii.w;
    if sun_radius <= 0.0 { return vec3(0.0); }
    let cos_angle = dot(ray_dir_ws, sun_dir);
    let angle = acos(clamp(cos_angle, -1.0, 1.0));
    let w = max(0.5 * fwidth(angle), 1e-6);
    let edge_mask = 1.0 - smoothstep(sun_radius - w, sun_radius + w, angle);

    // Limb darkening inside the disc (Allen 1973 for Sun-like stars).
    var disc_contribution = vec3(0.0);
    if edge_mask > 0.0 {
        let r_norm = saturate(angle / sun_radius);
        let mu = sqrt(max(1.0 - r_norm * r_norm, 0.0));
        let u1 = 0.56;
        let u2 = 0.16;
        let one_minus_mu = 1.0 - mu;
        let limb = 1.0 - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu;
        let flux_correction = 1.0 / (1.0 - u1 / 3.0 - u2 / 6.0);
        let solid_angle = PI * sun_radius * sun_radius;
        disc_contribution = atmo.sun_color.xyz * edge_mask * limb * flux_correction / max(solid_angle, 1e-6);
    }

    // Atmospheric corona: tight bright halo immediately around the disc only.
    // The broad wide glow was over-bright in space and competed with sky
    // inscatter near the sun — kept the tight ring for readable limb contrast
    // and dropped the broad tier.
    let outside = max(angle - sun_radius, 0.0);
    let tight_halo = exp(-outside * 400.0) * 0.3;
    let chroma = vec3(1.0, 0.95, 0.88);
    let halo_contribution = atmo.sun_color.xyz * chroma * tight_halo / max(PI * sun_radius * sun_radius, 1e-6);

    return disc_contribution + halo_contribution;
}

// ---------------------------------------------------------------------------
// Volumetric cloud ray march (unchanged from pre-LUT implementation).
// Density: Perlin-Worley base + Worley detail erosion, per-planet weather map.
// Lighting: Beer-Lambert + multi-scatter octave approximation.
// ---------------------------------------------------------------------------

fn remap_cloud(value: f32, old_lo: f32, old_hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + saturate((value - old_lo) / (old_hi - old_lo)) * (new_hi - new_lo);
}

fn cloud_height_gradient(height_frac: f32, type_blend: f32) -> f32 {
    let stratus = smoothstep(0.0, 0.15, height_frac) * (1.0 - smoothstep(0.15, 0.4, height_frac));
    let cumulus = smoothstep(0.0, 0.2, height_frac) * (1.0 - smoothstep(0.4, 0.95, height_frac));
    let cb = smoothstep(0.0, 0.15, height_frac) * (1.0 - smoothstep(0.6, 1.0, height_frac));
    return mix(
        mix(stratus, cumulus, saturate(type_blend * 2.0)),
        cb,
        saturate((type_blend - 0.5) * 2.0)
    );
}

fn sample_weather(pos: vec3<f32>) -> vec4<f32> {
    let dir = normalize(pos);
    let lon = atan2(dir.z, dir.x);
    let lat = asin(clamp(dir.y, -1.0, 1.0));
    let uv = vec2(lon / PI_2 + 0.5, lat / PI + 0.5);
    return textureSample(weather_map, cloud_sampler, uv);
}

fn cloud_density(pos: vec3<f32>) -> f32 {
    let planet_r = clouds.geometry.x;
    let cloud_base = clouds.geometry.y;
    let cloud_thick = clouds.geometry.z;
    let altitude = length(pos) - planet_r;
    let height_frac = (altitude - cloud_base) / cloud_thick;
    if height_frac <= 0.0 || height_frac >= 1.0 { return 0.0; }

    let weather = sample_weather(pos);
    let coverage = weather.r;
    if coverage < 0.01 { return 0.0; }
    let cloud_type = weather.g;
    let gradient = cloud_height_gradient(height_frac, cloud_type);
    if gradient < 0.001 { return 0.0; }

    let time = clouds.observer_pos.w;
    let wind_offset = clouds.wind.xyz * time;
    let shape_scale = clouds.noise_params.x;
    let shape_uv = (pos + wind_offset) * shape_scale;
    let shape = textureSample(cloud_shape_tex, cloud_sampler, shape_uv);
    let worley_fbm = shape.g * 0.625 + shape.b * 0.25 + shape.a * 0.125;
    let base_shape = remap_cloud(shape.r, worley_fbm - 1.0, 1.0, 0.0, 1.0);
    if base_shape < 0.05 { return 0.0; }

    var density = base_shape * gradient;
    density = remap_cloud(density, 1.0 - coverage, 1.0, 0.0, 1.0) * coverage;
    if density <= 0.0 { return 0.0; }

    let detail_scale = clouds.noise_params.y;
    let sheared_wind = wind_offset * clouds.wind.w;
    let detail_uv = (pos + sheared_wind) * detail_scale;
    let detail = textureSample(cloud_detail_tex, cloud_sampler, detail_uv);
    let detail_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
    let detail_mod = mix(detail_fbm, 1.0 - detail_fbm, saturate(height_frac * 2.0));
    density = remap_cloud(density, detail_mod * 0.2, 1.0, 0.0, 1.0);
    return max(density, 0.0) * clouds.density_params.y;
}

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

// Proper sun-direction shadow march: sample cloud density at two offsets along
// the sun ray from the current cloud sample, accumulate optical depth, apply
// Beer-Lambert. This is what creates visible volumetric god rays — thick cloud
// along the sun ray blocks direct sunlight, thin cloud lets it through in a
// beam pattern.
//
// Two samples at 0.5 and 1.5 km hit the sweet spot for cumulus/cumulonimbus
// scale (typical cloud features are 0.5–3 km across) without the cost of a
// full N-sample march. Combined with the Nubis multi-octave approximation of
// internal cloud scattering, this produces AAA-quality cloudscape lighting.
fn cloud_sun_shadow(pos: vec3<f32>, sun_dir: vec3<f32>, absorption: f32) -> f32 {
    let d0 = cloud_density(pos + sun_dir * 0.5);
    let d1 = cloud_density(pos + sun_dir * 1.5);
    // Effective optical depth over the 2 km sun path.
    let tau = (d0 + d1) * absorption * 1.0;
    return exp(-tau);
}

struct CloudResult {
    luminance: vec3<f32>,
    transmittance: f32,
    // Luminance-weighted mean distance (km) along the ray to the cloud contribution.
    // Used to apply aerial perspective: distant clouds should fade into the
    // sky color rather than remain at their local luminance.
    mean_distance_km: f32,
}

fn cloud_ray_march(ray_origin: vec3<f32>, ray_dir: vec3<f32>, max_dist: f32) -> CloudResult {
    var result: CloudResult;
    result.luminance = vec3(0.0);
    result.transmittance = 1.0;
    result.mean_distance_km = 0.0;
    var lum_weight_sum = 0.0;

    let planet_r = clouds.geometry.x;
    let cloud_base = clouds.geometry.y;
    let cloud_thick = clouds.geometry.z;
    let shell_inner = planet_r + cloud_base;
    let shell_outer = planet_r + cloud_base + cloud_thick;

    let t_inner = ray_sphere_both(ray_origin, ray_dir, shell_inner);
    let t_outer = ray_sphere_both(ray_origin, ray_dir, shell_outer);

    var t_start = 0.0;
    var t_end = 0.0;
    let obs_height = length(ray_origin);
    if obs_height < shell_inner {
        t_start = max(t_inner.x, 0.0);
        if t_inner.x < 0.0 { t_start = 0.0; }
        t_end = t_outer.y;
    } else if obs_height > shell_outer {
        if t_outer.x < 0.0 { return result; }
        t_start = t_outer.x;
        t_end = t_outer.y;
        if t_inner.x > 0.0 { t_end = min(t_end, t_inner.x); }
    } else {
        t_start = 0.0;
        t_end = t_outer.y;
        if t_inner.x > 0.0 { t_end = min(t_end, t_inner.x); }
    }

    t_end = min(t_end, max_dist);
    if t_start >= t_end || t_end <= 0.0 { return result; }

    let ray_length = t_end - t_start;
    let max_steps = 64u;
    let dt = ray_length / f32(max_steps);

    let pixel = vec2<f32>(
        fract(ray_origin.x * 1000.0 + ray_dir.x * 500.0),
        fract(ray_origin.y * 1000.0 + ray_dir.y * 500.0)
    ) * 1000.0;
    let jitter = fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));

    let cos_angle = dot(ray_dir, clouds.sun.xyz);
    let phase = max(hg_phase(cos_angle, 0.8), hg_phase(cos_angle, -0.5));
    let absorption = clouds.density_params.w;
    let sun_color_val = clouds.sun_color.xyz * clouds.sun.w;
    let scatter_color = clouds.scatter.xyz;

    for (var i = 0u; i < max_steps; i++) {
        let t = t_start + (f32(i) + jitter) * dt;
        let sample_pos = ray_origin + ray_dir * t;
        let density = cloud_density(sample_pos);
        if density > 0.001 {
            let sample_transmittance = exp(-density * absorption * dt);
            let sun_dir = clouds.sun.xyz;
            let cloud_normal = normalize(sample_pos);
            let sun_angle = dot(cloud_normal, sun_dir);
            let planet_shadow = smoothstep(-0.1, 0.15, sun_angle);

            var direct_light = vec3(0.0);
            if planet_shadow > 0.001 {
                // Proper sun-direction shadow march for god rays.
                let sun_shadow = cloud_sun_shadow(sample_pos, sun_dir, absorption);
                // Multi-scattering octaves: approximates how light bounces
                // inside the cloud. Each octave is a softer, more diffuse
                // contribution. Stacked with the real sun shadow march,
                // this gives dense cumulus its characteristic lit/shadow
                // contrast while preserving inter-cloud light beams.
                var oct_atten = 1.0;
                var oct_contrib = 1.0;
                var oct_phase = phase;
                for (var oct = 0u; oct < 3u; oct++) {
                    let ext = density * oct_atten * absorption;
                    let powder = 1.0 - exp(-ext * dt * 12.0);
                    direct_light += planet_shadow * sun_shadow * mix(0.7, 1.0, powder) * oct_phase * oct_contrib * scatter_color * sun_color_val;
                    oct_atten *= 0.5;
                    oct_contrib *= 0.5;
                    oct_phase = mix(oct_phase, 1.0 / (4.0 * PI), 0.5);
                }
            }

            let extinction = density * absorption;
            let direct_integrated = (direct_light - direct_light * sample_transmittance) / max(extinction, 0.001);
            let ambient = scatter_color * 0.008 * planet_shadow * planet_shadow * (1.0 - sample_transmittance);

            let contribution = result.transmittance * (direct_integrated + ambient);
            let contrib_mag = dot(contribution, vec3(0.299, 0.587, 0.114));
            result.luminance += contribution;
            result.mean_distance_km += t * contrib_mag;
            lum_weight_sum += contrib_mag;
            result.transmittance *= sample_transmittance;
            if result.transmittance < 0.01 { break; }
        }
    }

    if lum_weight_sum > 1e-6 {
        result.mean_distance_km = result.mean_distance_km / lum_weight_sum;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Screen-space procedural precipitation. Density and type driven by the
// weather map sampled at the observer's planet-relative direction. Only
// active when the observer is below the cloud layer (precipitation has to
// fall from somewhere). Streaks are deterministic from screen-cell hashing
// so they look consistent frame-to-frame while scrolling downward.
//
// Two modes blended by weather.G (cloud type): rain (thin slanted streaks)
// for mid-type clouds, snow (soft round flakes) when cloud type is very
// high and ambient cold. For initial implementation we always use rain;
// temperature/latitude-based snow is a Phase 5+ enhancement.
// ---------------------------------------------------------------------------

fn precip_hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn screen_space_rain(uv: vec2<f32>, aspect: f32, time: f32, density: f32, intensity: f32) -> f32 {
    // Tile screen into tall narrow cells. Aspect-correct the y so cells stay
    // vertical regardless of window shape; fall speed scales with cell height.
    let cell = vec2(0.018, 0.08);
    let uv_aspect = vec2(uv.x * aspect, uv.y);
    let cuv = uv_aspect / cell;
    let cell_id = floor(cuv);
    let cell_frac = fract(cuv);

    let h = precip_hash21(cell_id);
    // Active probability matches density; each cell independently rolled.
    if h < 1.0 - density { return 0.0; }

    // Streak falling speed: faster streaks closer to camera (higher h),
    // scroll via time. Add horizontal drift for wind feel.
    let speed = 1.0 + h * 2.5;
    let drift = (h - 0.5) * 0.3;
    let phase = time * speed + h * 10.0;
    let streak_x = 0.5 + drift;
    let streak_y = fract(phase);

    // Streak shape: narrow vertical line with soft edges; slight slant.
    let slant_shift = (cell_frac.y - 0.5) * 0.15;
    let dx = cell_frac.x - streak_x - slant_shift;
    let dy = cell_frac.y - streak_y;
    // Elongated ellipse: narrow x, long y.
    let shape = smoothstep(0.08, 0.0, abs(dx)) * smoothstep(0.25, 0.0, abs(dy));
    return shape * intensity;
}

// ---------------------------------------------------------------------------
// AgX display transform (Troy Sobotka). Port of the canonical "AgX Minimal"
// form — inset primaries → log encode → sigmoid polynomial → outset primaries.
// Beats ACES for saturation roll-off at extreme HDR values (sunrise sun disc,
// bright clouds near sun) and preserves natural-looking highlights.
// ---------------------------------------------------------------------------

const AGX_MAT: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.842479062253094,   0.0784335999999992,  0.0792237451477643),
    vec3<f32>(0.0423282422610123,  0.878468636469772,   0.0791661274605434),
    vec3<f32>(0.0423756549057051,  0.0784336,           0.879142973793104),
);
const AGX_MAT_INV: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>( 1.1968790051201738, -0.09802338718458979, -0.09902515035750428),
    vec3<f32>(-0.05290646989131923,  1.1519031299041727, -0.09896712436230928),
    vec3<f32>(-0.05297178517909099, -0.09804503613733328,  1.1518968062375834),
);

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx_tonemap(color_in: vec3<f32>) -> vec3<f32> {
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var c = AGX_MAT * max(color_in, vec3(0.0));
    c = clamp(log2(max(c, vec3(1e-10))), vec3(min_ev), vec3(max_ev));
    c = (c - min_ev) / (max_ev - min_ev);
    c = agx_default_contrast_approx(c);
    // "Punchy" look — mild saturation boost matching Blender's default AgX look.
    let lw = vec3(0.2126, 0.7152, 0.0722);
    let luma = dot(c, lw);
    c = luma + 1.2 * (c - luma);
    c = AGX_MAT_INV * c;
    return max(c, vec3(0.0));
}

// ---------------------------------------------------------------------------
// Fragment shader.
// ---------------------------------------------------------------------------

@fragment
fn fs_composite(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_scene, screen_sampler, in.uv);
    let hdr_color = hdr.rgb;
    let is_exterior = hdr.a; // 1.0 = exterior (atmosphere applies), 0.0 = interior
    let depth = textureLoad(depth_tex, vec2<i32>(in.position.xy), 0);

    // Reconstruct world-space ray direction (always; used for clouds + LUT lookup).
    let is_sky = depth < 1e-6;
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

    var color = hdr_color;

    // Atmosphere compositing — LUT-based.
    //
    // For observers above the atmosphere (r > outer_r), the transmittance and
    // sky-view LUTs are parameterized out of range. We detect this case and
    // split:
    //   - Ray misses the atmosphere entirely → transmittance = 1, no inscatter
    //     (this is the "pure space" case: stars and sun disc pass through cleanly).
    //   - Ray enters the atmosphere → we "shift" the effective observer to the
    //     atmosphere entry point along the ray so LUT sampling stays valid.
    if atmo.radii.z > 0.5 && is_exterior > 0.5 {
        let observer = atmo.observer_pos.xyz;
        let inner_r = atmo.radii.x;
        let outer_r = atmo.radii.y;
        let r_obs = length(observer);
        let in_atmo = r_obs <= outer_r;

        // Effective observer / r / mu for LUT lookups. When outside, advance
        // the observer to the atmosphere entry point; when inside, use as-is.
        var effective_observer = observer;
        var effective_r = max(r_obs, inner_r + 0.001);
        var ray_enters_atmo = in_atmo;
        if !in_atmo {
            let t_atmo = ray_sphere_both(observer, ray_dir, outer_r);
            // Ray enters the atmosphere only if it intersects in front of us.
            if t_atmo.y > 0.0 && t_atmo.x > 0.0 {
                effective_observer = observer + ray_dir * t_atmo.x;
                effective_r = outer_r - 0.001;
                ray_enters_atmo = true;
            } else {
                ray_enters_atmo = false;
            }
        }

        if is_sky {
            if ray_enters_atmo {
                let ray_as = (atmo.atmo_from_world * vec4(ray_dir, 0.0)).xyz;
                let inscatter = sample_skyview(effective_r, ray_as);
                let mu = dot(normalize(effective_observer), ray_dir);
                let trans = sample_transmittance_lut_rm(effective_r, mu);
                let sun_disc = sample_sun_disc(ray_dir);
                color = hdr_color * trans + sun_disc * trans + inscatter;
            } else {
                // Pure space: no atmospheric scattering, sun disc untouched.
                let sun_disc = sample_sun_disc(ray_dir);
                color = hdr_color + sun_disc;
            }
        } else {
            // Geometry pixels: aerial perspective if inside atmosphere;
            // otherwise no atmospheric modification (observer + geometry both
            // in space — the AP LUT wasn't built for this case).
            if in_atmo {
                let ap = sample_aerial(in.uv, frag_dist_km);
                let inscatter = ap.rgb;
                let mu = dot(normalize(observer), ray_dir);
                let trans = sample_transmittance_segment(effective_r, mu, frag_dist_km);
                color = color * trans + inscatter;
            }
        }
    }

    // Volumetric cloud compositing (Nubis). Distant clouds get aerial
    // perspective: atmospheric transmittance between the cloud and the camera
    // dims their luminance and shifts hue toward the sky, matching real-world
    // haze. Only applied when the atmosphere LUTs are valid.
    if clouds.geometry.w > 0.5 && is_exterior > 0.5 {
        let cloud_observer = clouds.observer_pos.xyz;
        let cloud_result = cloud_ray_march(cloud_observer, ray_dir, frag_dist_km);
        var cloud_lum = cloud_result.luminance;
        if atmo.radii.z > 0.5 && cloud_result.mean_distance_km > 0.0 {
            let ap = sample_aerial(in.uv, cloud_result.mean_distance_km);
            cloud_lum = cloud_lum * ap.a + ap.rgb;
        }
        color = color * cloud_result.transmittance + cloud_lum;
    }

    // Screen-space precipitation. Only render when below cloud layer,
    // exterior pixels, and weather says it's precipitating here.
    if clouds.geometry.w > 0.5 && is_exterior > 0.5 {
        let observer_pos = clouds.observer_pos.xyz;
        let planet_r = clouds.geometry.x;
        let cloud_top_r = planet_r + clouds.geometry.y + clouds.geometry.z;
        let observer_r = length(observer_pos);
        if observer_r < cloud_top_r {
            let weather = sample_weather(observer_pos);
            let precip = weather.b;
            if precip > 0.05 {
                let time = clouds.observer_pos.w;
                let aspect = params.screen_dims.x / max(params.screen_dims.y, 1.0);
                // Density scales with precip intensity; intensity controls
                // contrast so light rain is subtle and heavy storm is dense.
                let density = clamp(precip * 0.9, 0.05, 0.85);
                let rain = screen_space_rain(in.uv, aspect, time, density, precip);
                // Rain streak color: slightly bluish-grey, scaled by sun/ambient
                // so night-time rain stays visible but dim and day rain is bright.
                let sun_b = dot(atmo.sun_color.xyz, vec3(0.3)) + 0.1;
                color += vec3(0.55, 0.65, 0.8) * rain * sun_b * 0.6;
            }
        }
    }

    color = agx_tonemap(color);
    return vec4(color, 1.0);
}
