// THE STAR SPRITE (S11) — one camera-facing point of light per star, expanded on the GPU.
//
// ★ WHY A SHADER AT ALL. A baked quad gets three things wrong: it does not face the camera, so a star
// winks out as you fly past it; it cannot hold a minimum size, so a distant star falls below one pixel
// and flickers as it crosses pixel boundaries; and it shrinks with distance, which is backwards — a
// real star is a point source, so distance dims it rather than shrinking it.
//
// ★ EVERY CONSTANT ARRIVES AS A UNIFORM. Nothing here is a literal from the size law. The apparent-size
// floor lives in Tier-A (`vd_client_harness::camera::marker_world_radius`) and is shared with the pixel
// gates; a second copy typed into this file would let the GPU draw one rectangle while the gate asserts
// another, and both would look right on their own.
//
// ★ THIS SHADER DOES NOT CHOOSE A BRIGHTNESS FALLOFF. Brightness is a live owner knob with an undecided
// curve (D-WINDOW-3). Size carries luminosity, through the square-root convention that Tier-A already
// owns. Inventing a distance-brightness law here would be a decision taken by accident.

#import bevy_pbr::mesh_functions
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

struct StarSkyUniform {
    // The apparent-size floor, in pixels, from DOT_MIN_APPARENT_RADIUS_PX.
    min_apparent_radius_px: f32,
    // tan(fov_y / 2) — half the vertical field of view.
    tan_half_fov: f32,
    // The viewport height in rows.
    viewport_h_px: f32,
    // WHERE THE SKY SITS IN DEPTH, in metres — just inside the camera's far plane. See the vertex
    // stage: the star's true DIRECTION is kept, and only its depth slot is fixed.
    sky_radius_m: f32,
};

// ★ THE MATERIAL BIND GROUP, BY BEVY'S OWN SUBSTITUTION — never a literal index.
//
// A hard-coded `@group(2)` failed pipeline creation at runtime: group 2 is Bevy's MESH group and holds
// a STORAGE buffer, so wgpu refused with "Storage class Storage doesn't match the shader Uniform".
// `MATERIAL_BIND_GROUP` is 3 in this version and Bevy substitutes it, so a version that moves the group
// moves this with it. A literal here is a number that is right until it silently is not.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> star: StarSkyUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) base_radius_m: f32,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // The corner sign, carried through so the fragment stage can round the quad into a disc.
    @location(1) corner: vec2<f32>,
};

// THE WORLD SIZE OF ONE PIXEL at a view distance — a transliteration of Tier-A's `one_pixel_world_m`.
// A subject at or behind the eye has no forward extent, so the distance floors at zero.
fn one_pixel_world_m(dist_m: f32) -> f32 {
    return 2.0 * max(dist_m, 0.0) * star.tan_half_fov / star.viewport_h_px;
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    // The star's centre in view space. The sky rides an identity model transform: the positions are
    // already metres from the observer's own star system, which is where the camera stands.
    let view_pos = view.view_from_world * vec4<f32>(in.position, 1.0);
    let dist_m = length(view_pos.xyz);

    // ★ THE TRUE DIRECTION, AT A FIXED DEPTH — the sky's placement law.
    //
    // ★ MEASURED, AND LOAD-BEARING (2026-08-27). Drawn at its true view-space position a star at
    // 4.6e18 m paints NOTHING; placed at this radius it paints. The gate was run both ways to
    // establish that, after an earlier justification for this code turned out to be wrong.
    //
    // ⚠ THE MECHANISM IS NOT ESTABLISHED. The first explanation — that the camera's far plane clipped
    // it — is FALSE: this camera is a reverse-Z INFINITE perspective and has no far clip at all (see
    // the capture camera's own comment). The likely cause is depth or coordinate precision at 4.6e18 m
    // in f32, but that is a hypothesis, NOT a measurement, and it is written here as one. What is
    // measured is only the before and the after.
    //
    // WHAT THE PLACEMENT PRESERVES. The DIRECTION is kept exactly and only the depth slot is fixed, so
    // the angle a star subtends is real and parallax is real: move, and the angles change exactly as
    // true positions demand. That is what the owner's ruling requires — the star map shows the REAL
    // reachable systems, with no painted backdrop sphere and no second seed anywhere here.
    //
    // The true distance is NOT discarded: it still sizes the sprite through the apparent-size floor
    // below, which is what makes a far star small and a near one large.
    let dir = normalize(view_pos.xyz);
    let placed = dir * star.sky_radius_m;

    // THE APPARENT-SIZE FLOOR — Tier-A's `marker_world_radius`, evaluated per star on the GPU. On the
    // CPU this would be one distance computation per star per frame, which is exactly the per-frame
    // cost this whole design removes.
    let radius_m = max(
        in.base_radius_m,
        star.min_apparent_radius_px * one_pixel_world_m(dist_m)
    );

    // THE SPRITE'S SIZE AT ITS PLACED DEPTH. The radius above is a world size at the star's TRUE
    // distance; drawn at the sky radius it must be scaled by the same ratio, or a distant star would
    // balloon to the size it would have had if it really were that close.
    let placed_radius_m = radius_m * star.sky_radius_m / max(dist_m, 1.0);

    // BILLBOARD IN VIEW SPACE. Offsetting after the view transform makes the quad face the camera by
    // construction — there is no orientation to get wrong, and no branch.
    let offset = vec3<f32>(in.corner * placed_radius_m, 0.0);
    out.clip = view.clip_from_view * vec4<f32>(placed + offset, 1.0);
    out.color = in.color;
    out.corner = in.corner;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    // ROUND THE QUAD INTO A DISC, with a soft edge. A hard-edged square reads as a square the moment a
    // star is more than a pixel or two across, and aliases badly while the camera moves.
    let r = length(in.corner);
    // `fwidth` gives one pixel's worth of the radius here, so the falloff is one pixel wide whatever
    // the sprite's size on screen.
    let edge = fwidth(r);
    let alpha = 1.0 - smoothstep(1.0 - edge, 1.0, r);
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
