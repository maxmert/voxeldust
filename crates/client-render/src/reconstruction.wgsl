// ★ THE RECONSTRUCTION (the temporal path's slice T2; `docs/investigation/2026-09-28/temporal_path_design.md`
// §4.2): each DISPLAY pixel rebuilt from this frame's jittered RENDER samples and the history of the frames
// before, on the structure of AMD FSR 2.2 (MIT). The rules are `vd_client::reconstruction`'s, step for step:
// the sample filter (Lanczos-2 of each sample's distance, narrowed by the scale and widened for a pixel that
// starts again), the de-ringing (the colour clamped into the samples' range), the frame's EARNED weight (the
// sum of its samples' weights; a display pixel between samples earns little and keeps its history), the count
// of earned weight capped by motion (MOTION ALONE NEVER RESTARTS THE COUNT), the tone weight (Karis 2014), and
// the history's rectification into the colour box in YCoCg (FSR 2.2's `RectifyHistory`: a clamp, axis by axis).
// Adapted from Bevy's TAA (MIT; its history filter, its reversible tone map).
//
// The render picture sits in the top-left `render_size` rectangle of full-size textures (the engine's own
// resolution override); render pixel `i` samples the scene at `i + 0.5 + jitter`, in render pixels (the
// engine's projection offset draws a scene point at `p` as `p − offset`, so the pixel centre `c` sees
// `c + offset`; `vd_client::reconstruction::jitter` states the sign, and a test of the engine's own
// projection proves it). FSR 2 writes the same geometry as `c − jitter` with its jitter of the opposite
// sign. The motion is in UV units of the picture (this frame's place less last frame's), so it serves the
// render grid and the display grid alike.
//
// ★ THE DEPTH TEST (the design §4.2 steps 1.3 and 2; FSR 2.2's `ReconstructPrevDepth` and `ComputeDepthClip`):
// the compute entry `rebuild` writes each render pixel's nearest depth at its last-frame place, and the
// fragment finds a pixel that a nearer surface hid in the last frame — a disocclusion — and lowers its count.
//
// ★ THE REACTIVE MASK (the design §4.2 step 3; FSR 2's automatic reactive mask): the render picture as it stood
// before the transparent pass (a copy), against the picture after it — a pixel a translucent thing changed takes
// mostly the new frame, and its colour box shrinks.
//
// The sky pixels reproject by the two views turned by the air's rotation step (the design §6.3), which the
// render world folds into `sky_reproject`.
//
// NOT YET: FSR 2's sample-closeness weight.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct Params {
    // Last frame's unjittered clip from this frame's unjittered clip, through a point at infinity: the
    // motion of a pixel that no surface wrote (the sky).
    sky_reproject: mat4x4<f32>,
    render_size: vec2<f32>,
    display_size: vec2<f32>,
    // Where each render pixel's sample lies this frame, from its centre, in render pixels
    // (`vd_client::reconstruction::jitter`).
    jitter: vec2<f32>,
    // One on the frame the history restarts (`vd_client::reconstruction::reset_cause`).
    reset: u32,
    // The motion target's clear (`vd_client_harness::motion_census::MOTION_SENTINEL`): no surface wrote it.
    motion_sentinel: f32,
    // `vd_client::reconstruction::ReconstructionSettings`.
    still_floor: f32,
    moving_floor: f32,
    still_motion_px: f32,
    moving_motion_px: f32,
    box_deviations: f32,
    // The filter's narrowing (`vd_client::reconstruction::kernel_bias`): the render offset times this.
    kernel_bias: f32,
    // The depth test (`vd_client::reconstruction`): the projection's near plane (a depth is `near ÷ depth`
    // metres along the view), the smallest depth divided by, the separation per metre of depth, the
    // test's power, the still motion in display pixels, and the bilinear tap's floor.
    near: f32,
    depth_floor: f32,
    depth_separation: f32,
    depth_clip_power: f32,
    rebuild_still_display_px: f32,
    bilinear_tap_floor: f32,
    // The reactive mask (`vd_client::reconstruction::reactive`): the change's scale, its threshold, the value.
    reactive_scale: f32,
    reactive_threshold: f32,
    reactive_value: f32,
    // A frame's earned weight under this counts as none (`ReconstructionSettings::upsample_weight_min`).
    upsample_weight_min: f32,
}

@group(0) @binding(0) var colour_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d<f32>;
@group(0) @binding(2) var count_tex: texture_2d<f32>;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var<uniform> params: Params;
// ★ THE REBUILT PREVIOUS DEPTH, one word per render pixel of the last frame's grid: the nearest of this
// frame's surfaces that stood there, as the bits of its reverse-Z depth (a larger depth is nearer, and the
// bits of a positive float order as the float); nought where no surface stood. The compute entry writes it,
// the fragment reads it.
@group(0) @binding(7) var<storage, read_write> rebuilt_write: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> rebuilt_read: array<u32>;
// The render picture before the transparent pass (the reactive mask's copy, the render rectangle only).
@group(0) @binding(9) var opaque_tex: texture_2d<f32>;

struct Output {
    // The display picture: linear, alpha ONE (the design §7.7, change two).
    @location(0) colour: vec4<f32>,
    // The new history: the tone-mapped colour, and the sky share in the fourth channel (§7.7).
    @location(1) history: vec4<f32>,
    // The new count of samples.
    @location(2) count: vec4<f32>,
}

const PI: f32 = 3.14159265358979;

fn max3(x: vec3<f32>) -> f32 {
    return max(x.r, max(x.g, x.b));
}

// THE TONE WEIGHT (Karis 2014; `vd_client::reconstruction::tone_weight`), as a reversible map: a bright
// point counts as one, so it cannot dominate a pixel's sum. A picture without HDR is already in [0, 1].
fn tonemap(c: vec3<f32>) -> vec3<f32> {
#ifdef TONEMAP
    return c / (max3(c) + 1.0);
#else
    return c;
#endif
}

fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> {
#ifdef TONEMAP
    return c / max(1.0 - max3(c), 1.0e-4);
#else
    return c;
#endif
}

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// THE SAMPLE FILTER (`vd_client::reconstruction::lanczos2`).
fn sinc(x: f32) -> f32 {
    let p = PI * x;
    return select(sin(p) / p, 1.0, abs(p) < 1.0e-6);
}

fn lanczos2(x: f32) -> f32 {
    return select(0.0, sinc(x) * sinc(x * 0.5), abs(x) < 2.0);
}

// THE FLOOR BY MOTION (`vd_client::reconstruction::weight_floor`): a line in the motion's logarithm.
fn weight_floor(motion_px: f32) -> f32 {
    let lo = log(params.still_motion_px);
    let hi = log(params.moving_motion_px);
    let t = clamp((log(max(motion_px, 1.0e-30)) - lo) / (hi - lo), 0.0, 1.0);
    return mix(params.still_floor, params.moving_floor, t);
}

// ★ THE HISTORY'S RECTIFICATION (FSR 2.2's `RectifyHistory`; `vd_client::reconstruction::rectify`): the history
// clamped, axis by axis in YCoCg, into the box of `centre ± extent` cut to the samples' own range — the box's
// nearest point, which never carries two colours apart. The clip toward the centre it replaces (2026-10-01,
// ruling B28 item 3) carried them apart on a slope, and a world a few ticks away drew the hill's dunes 7 levels
// off. A box that rounding inverts takes its upper bound.
fn rectify(history: vec3<f32>, centre: vec3<f32>, extent: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let lower = max(centre - extent, lo);
    let upper = min(centre + extent, hi);
    return min(max(history, lower), upper);
}

// ★ THE MOTION of the surface at render pixel `at` (depth `z`), in UV units: the prepass's; where no surface
// wrote one (the sky, the sentinel), the two views' motion through a point at infinity at `uv`.
fn motion_at(at: vec2<i32>, z: f32, uv: vec2<f32>) -> vec2<f32> {
    let written = textureLoad(motion_tex, at, 0).xy;
    if (z == 0.0) || (written.x >= params.motion_sentinel * 0.5) {
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let previous = params.sky_reproject * vec4<f32>(ndc, 0.0, 1.0);
        return (ndc - previous.xy / previous.w) * vec2<f32>(0.5, -0.5);
    }
    return written;
}

// THE BILINEAR FOOTPRINT (`vd_client::reconstruction::bilinear_footprint`): the lower corner pixel of a point
// on the render grid, and the weights of its taps (0, 0), (1, 0), (0, 1), (1, 1) — tap `k` at the corner plus
// `(k & 1, k >> 1)`.
struct Footprint {
    base: vec2<i32>,
    weights: vec4<f32>,
}

fn footprint(at: vec2<f32>) -> Footprint {
    let corner = at - 0.5;
    let base = floor(corner);
    let f = corner - base;
    return Footprint(
        vec2<i32>(base),
        vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y),
    );
}

// The word of render pixel `q` in the rebuilt depth, if `q` lies on the render grid.
fn rebuilt_index(q: vec2<i32>) -> i32 {
    let size = vec2<i32>(params.render_size);
    return select(-1, q.y * size.x + q.x, all(q >= vec2<i32>(0)) && all(q < size));
}

// ★ THE REBUILT PREVIOUS DEPTH (the design §4.2 step 1.3; FSR 2.2's `ReconstructPrevDepth`), one invocation
// per render pixel: the nearest surface of its 3 × 3 and that surface's motion (a silhouette moves with its
// foreground), then its depth written at its last-frame place — into every tap of the place's bilinear
// footprint over the floor — keeping the nearest. The sky hides nothing and writes nothing.
@compute @workgroup_size(8, 8, 1)
fn rebuild(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(params.render_size);
    let p = vec2<i32>(id.xy);
    if any(p >= size) {
        return;
    }
    var depth = 0.0;
    var at = p;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), size - 1);
            let z = textureLoad(depth_tex, q, 0);
            if z > depth {
                depth = z;
                at = q;
            }
        }
    }
    if depth <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(p) + 0.5) / params.render_size;
    var motion = motion_at(at, depth, uv);
    // A motion under the still share of a display pixel is the reprojection's noise (FSR 2.2).
    motion *= f32(length(motion * params.display_size) > params.rebuild_still_display_px);
    let fp = footprint((uv - motion) * params.render_size);
    for (var k = 0; k < 4; k++) {
        let index = rebuilt_index(fp.base + vec2<i32>(k & 1, k >> 1));
        if (index >= 0) && (fp.weights[k] > params.bilinear_tap_floor) {
            atomicMax(&rebuilt_write[index], bitcast<u32>(depth));
        }
    }
}

// ★ THE DEPTH TEST (the design §4.2 step 2; FSR 2.2's `ComputeDepthClip`; `vd_client::reconstruction::
// depth_clip`): how much of this pixel a nearer surface hid in the last frame, from the rebuilt depth at its
// last-frame place `previous_px` (render pixels) and its own depth `current`.
fn depth_clip(previous_px: vec2<f32>, current: f32) -> f32 {
    let fp = footprint(previous_px);
    let here = params.near / max(current, params.depth_floor);
    var kept = 0.0;
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let index = rebuilt_index(fp.base + vec2<i32>(k & 1, k >> 1));
        let w = fp.weights[k];
        if (index >= 0) && (w > params.bilinear_tap_floor) {
            let before = params.near / max(bitcast<f32>(rebuilt_read[index]), params.depth_floor);
            if here > before {
                let share = saturate(params.depth_separation * max(here, before) / (here - before));
                kept += pow(share, params.depth_clip_power) * w;
                weight += w;
            }
        }
    }
    return select(0.0, saturate(1.0 - kept / weight), weight > 0.0);
}

// The history at `uv` by Catmull-Rom (five taps, Bevy's TAA after MJP and Activision): sharp, no blur.
fn history_at(uv: vec2<f32>) -> vec4<f32> {
    let size = params.display_size;
    let texel = 1.0 / size;
    let position = uv * size;
    let centre = floor(position - 0.5) + 0.5;
    let f = position - centre;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let p0 = (centre - 1.0) * texel;
    let p3 = (centre + 2.0) * texel;
    let p12 = (centre + w2 / w12) * texel;
    var h = textureSampleLevel(history_tex, linear_sampler, vec2(p12.x, p0.y), 0.0) * w12.x * w0.y;
    h += textureSampleLevel(history_tex, linear_sampler, vec2(p0.x, p12.y), 0.0) * w0.x * w12.y;
    h += textureSampleLevel(history_tex, linear_sampler, vec2(p12.x, p12.y), 0.0) * w12.x * w12.y;
    h += textureSampleLevel(history_tex, linear_sampler, vec2(p3.x, p12.y), 0.0) * w3.x * w12.y;
    h += textureSampleLevel(history_tex, linear_sampler, vec2(p12.x, p3.y), 0.0) * w12.x * w3.y;
    let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return h / weight;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> Output {
    let display_px = in.position.xy;
    let uv = display_px / params.display_size;
    let render_pos = display_px * (params.render_size / params.display_size);
    let last = vec2<i32>(params.render_size) - 1;
    // Render pixel `i` samples the scene at `i + 0.5 + jitter`: the sample nearest this display pixel's centre
    // (`vd_client::reconstruction::nearest_sample`).
    let nearest = vec2<i32>(floor(render_pos - params.jitter));

    // ★ THE SAMPLES (the design §4.2 step 4.1): the 3 × 3 around the nearest one (index 4), their tone, their sky
    // share, their offsets from this display pixel's centre, whether they lie on the render picture; their range
    // (the de-ringing's clamp) and their colour box; the nearest surface for the motion.
    var tones: array<vec3<f32>, 9>;
    var shares: array<f32, 9>;
    var offsets: array<vec2<f32>, 9>;
    var on_picture: array<f32, 9>;
    var lo = vec3<f32>(3.0e38);
    var hi = vec3<f32>(0.0);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    // The samples' own range in YCoCg: the rectification box's cut.
    var ylo = vec3<f32>(3.0e38);
    var yhi = vec3<f32>(-3.0e38);
    var closest_depth = 0.0;
    var closest = clamp(nearest, vec2<i32>(0), last);
    // ★ THE REACTIVE VALUE, the largest of the 3 × 3 (FSR 2 dilates its mask the same way).
    var reactive = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let k = (dy + 1) * 3 + (dx + 1);
            let raw = nearest + vec2<i32>(dx, dy);
            let q = clamp(raw, vec2<i32>(0), last);
            let sample = textureLoad(colour_tex, q, 0);
            let tone = tonemap(max(sample.rgb, vec3<f32>(0.0)));
            tones[k] = tone;
            shares[k] = 1.0 - sample.a;
            // `vd_client::reconstruction::sample_offset`, from the sample's own place: off the picture it earns
            // nothing.
            offsets[k] = vec2<f32>(raw) + 0.5 + params.jitter - render_pos;
            on_picture[k] = f32(all(raw == q));
            lo = min(lo, tone);
            hi = max(hi, tone);
            let y = to_ycocg(tone);
            m1 += y;
            m2 += y * y;
            ylo = min(ylo, y);
            yhi = max(yhi, y);
            let before = tonemap(max(textureLoad(opaque_tex, q, 0).rgb, vec3<f32>(0.0)));
            let change = max3(abs(tone - before)) * params.reactive_scale;
            reactive = max(reactive, select(0.0, params.reactive_value, change >= params.reactive_threshold));
            let z = textureLoad(depth_tex, q, 0);
            if z > closest_depth {
                closest_depth = z;
                closest = q;
            }
        }
    }
    let mean = m1 / 9.0;
    let deviation = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));

    // ★ THE MOTION: the nearest surface's (a silhouette moves with its foreground); where no surface wrote one
    // (the sky, the sentinel), from the two views through a point at infinity.
    let motion = motion_at(closest, closest_depth, uv);
    let history_uv = uv - motion;
    let motion_px = length(motion * params.render_size);

    // ★ THE RESTART AND THE DEPTH TEST: a reset or a pixel from off the screen starts again; a surface hidden in
    // the last frame lowers the count.
    let off_screen = any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0));
    let restart = (params.reset != 0u) || off_screen;
    let history_px = clamp(vec2<i32>(history_uv * params.display_size), vec2<i32>(0), vec2<i32>(params.display_size) - 1);
    let clip = depth_clip(history_uv * params.render_size, closest_depth);

    // ★ THE UPSAMPLE (FSR 2.2's with its de-ringing; `vd_client::reconstruction::{pixel_kernel_bias, upsample}`):
    // the filter narrowed by the scale and widened for a pixel that starts again, was hidden or is reactive; each
    // sample by Lanczos-2 of its biased DISTANCE; the colour clamped into the samples' range, so the filter's
    // negative lobes can never overshoot an edge; and the frame's EARNED weight, the weights' sum. MEASURED
    // (2026-09-30): the first build normalized by that sum and drew pixels brighter than the sky along the dune's
    // edge, where a display pixel between a dune and a haze sample earned a few thousandths.
    let widen = saturate(max(max(0.25 * clip, reactive), f32(restart)));
    let widest = max(1.0, (1.0 + params.kernel_bias) * 0.3);
    let bias = params.kernel_bias + (widest - params.kernel_bias) * widen;
    var sum = vec3<f32>(0.0);
    var share_sum = 0.0;
    var earned_sum = 0.0;
    for (var k = 0; k < 9; k++) {
        let w = on_picture[k] * lanczos2(length(offsets[k] * bias));
        sum += tones[k] * w;
        share_sum += shares[k] * w;
        earned_sum += w;
    }
    let gathered = earned_sum > params.upsample_weight_min;
    let divisor = max(earned_sum, params.upsample_weight_min);
    let current = select(tones[4], clamp(sum / divisor, lo, hi), gathered);
    let share = select(shares[4], saturate(share_sum / divisor), gathered);
    let earned = select(0.0, earned_sum, gathered);

    // ★ THE COUNT AND THE BLEND (the design §4.2 step 4.4; `vd_client::reconstruction::{next_count, count_cap,
    // blend_weight}`): the history's count, lowered by the stronger of the depth test and the reactive value (to
    // nought on a restart), plus the earned weight, capped by motion; the frame blends by its earned weight over
    // that count — a frame that earned nothing leaves the pixel to its history.
    let lowering = max(clip, reactive);
    let kept = select(textureLoad(count_tex, history_px, 0).r * (1.0 - lowering), 0.0, restart);
    let n = min(kept + earned, 1.0 / weight_floor(motion_px));
    let weight = select(1.0, saturate(earned / n), n > 0.0);

    // ★ THE HISTORY, RECTIFIED into the box of the new samples (the mean ± the settings' deviations, cut to the
    // samples' own range, in YCoCg); the reactive value shrinks the box.
    let history = history_at(history_uv);
    let extent = deviation * params.box_deviations * (1.0 - reactive);
    let clipped = from_ycocg(rectify(to_ycocg(history.rgb), mean, extent, ylo, yhi));

    let rgb = mix(max(clipped, vec3<f32>(0.0)), current, weight);
    // ★ A SHARE IS A SHARE, IN [0, 1]: the history's sharp filter dips under nought and rises over one at a
    // hard edge (Catmull-Rom's negative lobes). MEASURED (2026-09-30): at the edge of the home star's opaque
    // marker inside the sun, the share read a little under nought, the points pass multiplied the sun's disc by
    // it, and three pixels went black — the owner's "two black circles" in the sun.
    let sky_share = saturate(mix(history.a, share, weight));
    var out: Output;
    out.colour = vec4<f32>(reverse_tonemap(rgb), 1.0);
    out.history = vec4<f32>(rgb, sky_share);
    out.count = vec4<f32>(n, 0.0, 0.0, 1.0);
    return out;
}
