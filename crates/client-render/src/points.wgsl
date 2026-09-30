// ★ THE POINTS PASS (the temporal path's slice T2; `docs/investigation/2026-09-28/temporal_path_design.md`
// §7.7; ruling B24): the sun's disc and the star sprites drawn AFTER the reconstruction, at the DISPLAY size,
// added to the display picture before bloom and the tone map.
//
// WHY. They are the brightest and the sharpest things in the picture, and a star's core (σ 0.6 display
// pixels) is under one render pixel at the scales that ship: a jittered render picture holds it in some
// frames and not in others, so the history's clip deletes it (Karis 2014). Drawn here, a sprite is exactly
// today's sprite (the one law, `star_law.wgsl`; its sizes are display pixels, F24), never jittered.
//
// EACH POINT IS DIMMED BY TWO FACTORS:
// 1. the air's transmittance along its own direction, from the air's transmittance table (the engine builds
//    it every frame in both air modes) — where today the air pass multiplies the stars by its own per-pixel
//    transmittance (the design's difference 1: the table's 40 samples a ray, not the pixel's march);
// 2. the SKY SHARE at its display pixel: one less the render picture's alpha (the clear alpha is nought, a
//    surface writes one, a translucent box blends "over"), accumulated by the reconstruction with the colour
//    and read from its new history's fourth channel — so a star sets behind the belt's ridge by the ridge's
//    true accumulated coverage, smoothly, and never blinks with the jitter.
//
// The air pass no longer adds the sun on sky pixels under this path (the vendored change of T2): the disc is
// drawn here, once.
//
// Example: at the dusk stand a star sets behind the belt's ridge. It dims over the last display pixel and goes
// out; it never flickers in and out with the jitter.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import vd_client_render::star_law::{
    StarSkyUniform, one_pixel_world_m, star_amplitude, star_crop_px, star_profile,
}
#ifdef AIR
#import bevy_pbr::atmosphere::{
    bindings::{view, atmosphere},
    functions::{
        get_view_position, get_eye_position, sample_sun_radiance, sample_transmittance_lut,
        below_transmittance,
    },
}
#endif

struct Points {
    // The camera's UNJITTERED projection and its inverse: the points never take the jitter.
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    // The view's rotation and position (world from view), and its inverse (view from world).
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    // The star cloud's placement: its entity's model transform (the anchor and the eye, folded in f64).
    world_from_local: mat4x4<f32>,
}

@group(0) @binding(20) var<uniform> points: Points;
@group(0) @binding(21) var<uniform> star: StarSkyUniform;
// The reconstruction's new history: the sky share in its fourth channel (the design §7.7).
@group(0) @binding(22) var history_tex: texture_2d<f32>;

// ★ THE AIR ALONG A DIRECTION: the transmittance from the eye to the top of the air, from the table; one with
// no air. An eye under the air's ground sphere (over a dry sea floor) also looks through the air between it
// and the sphere (the vendored "air under its ground", 2026-09-30).
fn air_transmittance(ray_ws: vec3<f32>) -> vec3<f32> {
#ifdef AIR
    let eye = get_view_position();
    let r = length(eye);
    let mu = dot(ray_ws, normalize(eye));
    let true_eye = get_eye_position();
    return sample_transmittance_lut(r, mu)
        * below_transmittance(atmosphere, length(true_eye), dot(ray_ws, normalize(true_eye)));
#else
    return vec3<f32>(1.0);
#endif
}

// The sky share at a display pixel.
fn sky_share(pixel: vec2<f32>) -> f32 {
    return textureLoad(history_tex, vec2<i32>(pixel), 0).a;
}

// ★ THE SUN'S DISC, over the whole display picture: the air's own disc law (`sample_sun_radiance`: its angular
// size, its limb, the planet's shadow) along each display pixel's UNJITTERED ray, times the table's
// transmittance and the sky share. With no air there is no disc (the air pass drew none either).
@fragment
fn sun_fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
#ifdef AIR
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let at = points.view_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let ray_vs = normalize(at.xyz / at.w);
    let ray_ws = normalize((points.world_from_view * vec4<f32>(ray_vs, 0.0)).xyz);
    // ★ THE CAMERA'S EXPOSURE, as the sky pass applies it to its own light (its "exposure compensation"):
    // MEASURED (2026-09-30) without it, the disc stood about a thousand times too bright, its core past the
    // largest half float, and the tone map drew those pixels black — the owner's black circles in the sun.
    // (The star sprites never took the exposure: today's star material writes its light as it is.)
    let light = sample_sun_radiance(ray_ws) * air_transmittance(ray_ws) * view.exposure
        * sky_share(in.position.xy);
    return vec4<f32>(light, 0.0);
#else
    return vec4<f32>(0.0);
#endif
}

struct StarIn {
    @location(0) position: vec3<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) base_radius_m: f32,
};

struct StarOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // The offset from the star's centre in DISPLAY PIXELS (the sprite grown by half a pixel, below).
    @location(1) offset_px: vec2<f32>,
    @location(2) amplitude: f32,
    @location(3) crop_px: f32,
    // The air's transmittance along the star's direction (per sprite: it changes slowly across one).
    @location(4) transmittance: vec3<f32>,
};

// ★ TODAY'S SPRITE, EXACTLY: today's stars are drawn with four samples a pixel (the picture camera's MSAA), so a
// pixel takes the colour shaded ONCE at its centre times the share of its four samples inside the sprite — and
// a pixel whose centre lies outside the sprite still takes its covered share. MEASURED (a model of the law's own
// constants): for a faint star at the one-pixel crop, today's draw gives 0.922 of the law's exact light and a
// draw that counts whole pixels by their centres gives 0.997, so the milky band came out 8 % brighter than
// today's. This pass draws the sprite grown by half a pixel and counts the same four samples, the standard 4x
// positions (Direct3D, Metal and Vulkan), in pixels from the pixel's centre.
const SAMPLE_0: vec2<f32> = vec2<f32>(-0.125, -0.375);
const SAMPLE_1: vec2<f32> = vec2<f32>(0.375, -0.125);
const SAMPLE_2: vec2<f32> = vec2<f32>(-0.375, 0.125);
const SAMPLE_3: vec2<f32> = vec2<f32>(0.125, 0.375);
// The sprite's growth: every pixel that holds a covered sample holds its centre within half a pixel of the crop.
const GROWTH_PX: f32 = 0.5;

// One sample inside the sprite's square (the crop on each axis), as one or nought.
fn inside(offset_px: vec2<f32>, crop_px: f32) -> f32 {
    return f32(all(abs(offset_px) <= vec2<f32>(crop_px)));
}

// ★ THE STAR SPRITE, as the sky's own material draws it (`star_sky.wgsl`): the true direction at the sky's
// depth slot, sized by the law, billboarded in view space — through the UNJITTERED projection.
@vertex
fn star_vertex(in: StarIn) -> StarOut {
    var out: StarOut;
    let world_pos = points.world_from_local * vec4<f32>(in.position, 1.0);
    let view_pos = points.view_from_world * world_pos;
    let dist_m = length(view_pos.xyz);
    let dir = normalize(view_pos.xyz);
    let placed = dir * star.sky_radius_m;
    let amplitude = star_amplitude(star, in.base_radius_m, dist_m);
    let crop_px = star_crop_px(star, amplitude);
    // Grown by half a pixel, so every pixel with a covered sample is drawn; a culled star stays collapsed.
    let grown_px = select(0.0, crop_px + GROWTH_PX, crop_px > 0.0);
    let placed_radius_m = grown_px * one_pixel_world_m(star, star.sky_radius_m);
    let offset = vec3<f32>(in.corner * placed_radius_m, 0.0);
    out.clip = points.clip_from_view * vec4<f32>(placed + offset, 1.0);
    out.color = in.color;
    out.offset_px = in.corner * grown_px;
    out.amplitude = amplitude;
    out.crop_px = crop_px;
    out.transmittance = air_transmittance(normalize((points.world_from_view * vec4<f32>(dir, 0.0)).xyz));
    return out;
}

@fragment
fn star_fragment(in: StarOut) -> @location(0) vec4<f32> {
    // Shaded once, at the pixel's centre (today's rule, `star_sky.wgsl`: below one visible step, nothing).
    let intensity = in.amplitude * star_profile(star, length(in.offset_px));
    if (intensity <= star.cull_level) {
        discard;
    }
    let covered = 0.25 * (inside(in.offset_px + SAMPLE_0, in.crop_px)
        + inside(in.offset_px + SAMPLE_1, in.crop_px)
        + inside(in.offset_px + SAMPLE_2, in.crop_px)
        + inside(in.offset_px + SAMPLE_3, in.crop_px));
    if (covered == 0.0) {
        discard;
    }
    // TRUE ADDITION (the blend state adds the colour and keeps the picture's alpha).
    return vec4<f32>(in.color.rgb * intensity * covered * in.transmittance * sky_share(in.clip.xy), 0.0);
}
