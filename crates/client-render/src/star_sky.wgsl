// THE STAR SPRITE (S11) — one camera-facing point of light per star, expanded on the GPU.
//
// ★ WHY A SHADER AT ALL. A baked quad gets three things wrong: it does not face the camera, so a star
// winks out as you fly past it; it cannot hold a minimum size, so a distant star falls below one pixel
// and flickers as it crosses pixel boundaries; and it shrinks with distance, which is backwards — a
// real star is a point source, so distance dims it rather than shrinking it.
//
// ★ EVERY CONSTANT ARRIVES AS A UNIFORM, AND THE LAW IS ONE MODULE (`star_law.wgsl`): the size, the crop and
// the profile are shared with the points pass (`points.wgsl`, the temporal path's T2), so the star cloud's
// two draws cannot disagree. Nothing here is a literal from the size law; the apparent-size floor lives in
// Tier-A (`vd_client_harness::camera::marker_world_radius`) and is shared with the pixel gates.

#import bevy_pbr::mesh_functions
#import bevy_render::view::View
#import vd_client_render::star_law::{
    StarSkyUniform, one_pixel_world_m, star_amplitude, star_crop_px, star_profile,
}

@group(0) @binding(0) var<uniform> view: View;

// ★ THE MATERIAL BIND GROUP, BY BEVY'S OWN SUBSTITUTION — never a literal index.
//
// A hard-coded `@group(2)` failed pipeline creation at runtime: group 2 is Bevy's MESH group and holds
// a STORAGE buffer, so wgpu refused with "Storage class Storage doesn't match the shader Uniform".
// `MATERIAL_BIND_GROUP` is 3 in this version and Bevy substitutes it, so a version that moves the group
// moves this with it. A literal here is a number that is right until it silently is not.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> star: StarSkyUniform;

struct VertexIn {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) base_radius_m: f32,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // The corner sign, carried through so the fragment stage can shape the profile.
    @location(1) corner: vec2<f32>,
    // The profile's peak. Above 1.0 the core clips to white and the colour survives in the wings —
    // which is what a bright star looks like, arrived at by the physics rather than added on.
    @location(2) amplitude: f32,
    // The sprite's half-size in PIXELS, so the fragment stage can measure the profile in the same
    // unit its widths are stated in.
    @location(3) crop_px: f32,
};

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    // The star's centre in view space. The sky rides an identity model transform: the positions are
    // already metres from the observer's own star system, which is where the camera stands.
    // THE MODEL TRANSFORM IS THE SKY'S PLACEMENT (2026-09-02): the cloud's vertices are metres from
    // its reference cell, and the entity's transform carries them to the eye — the anchor the
    // gateway stated plus the eye's own offset, folded on the CPU in f64. This used to read the
    // vertex position as already eye-relative and ignored the transform entirely.
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_pos = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    let view_pos = view.view_from_world * world_pos;
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

    // ★ THE AMPLITUDE AND THE CROP, by the law (`star_law.wgsl`): the inverse-square flux from what the
    // vertex already carries, compressed as sight is, and the sprite cut where its wings fade below what
    // a screen can show — never under the minimum crop, and culled on brightness, never on size.
    let amplitude = star_amplitude(star, in.base_radius_m, dist_m);
    let crop_px = star_crop_px(star, amplitude);

    // THE SPRITE AT ITS PLACED DEPTH. `crop_px` is a size in PIXELS, so it converts through one
    // pixel's world size AT THE DEPTH THE SPRITE IS DRAWN — not at the star's true distance, which
    // would balloon a far star to the size it would have had if it were close.
    let placed_radius_m = crop_px * one_pixel_world_m(star, star.sky_radius_m);

    // BILLBOARD IN VIEW SPACE. Offsetting after the view transform makes the quad face the camera by
    // construction — there is no orientation to get wrong, and no branch.
    let offset = vec3<f32>(in.corner * placed_radius_m, 0.0);
    out.clip = view.clip_from_view * vec4<f32>(placed + offset, 1.0);
    out.color = in.color;
    out.corner = in.corner;
    out.amplitude = amplitude;
    out.crop_px = crop_px;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    // ★ A STAR IS A PEAK WITH WINGS, NOT A FILLED DISC (2026-08-29; the profile is the law's,
    // `star_law.wgsl`). This stage used to return `alpha = 1.0` across the whole interior with a one-pixel
    // soft rim — a flat plate, which together with the old 3-pixel floor is exactly what the owner saw:
    // 233 220 identical pieces of confetti.
    //
    // `in.corner` runs -1..1 across the quad, and the quad was cropped to `crop_px` — so the radius in
    // PIXELS is the corner times that crop, which is what the profile's sigmas are measured in.
    let r_px = length(in.corner) * (in.crop_px);
    let intensity = in.amplitude * star_profile(star, r_px);
    if (intensity <= star.cull_level) {
        // Below one visible step. Discarding here rather than drawing near-black keeps the overdraw
        // down, which matters at a quarter of a million sprites.
        discard;
    }

    // ★ PREMULTIPLIED, ALPHA ZERO — TRUE ADDITION. Bevy maps `AlphaMode::Add` to premultiplied-alpha
    // blending, whose colour term is `src + dst·(1 - src.a)`. With alpha 0 that is exactly `src + dst`,
    // so overlapping stars SUM instead of the nearer one replacing the further.
    //
    // ★ AND THE WHITE CORE ARRIVES FREE. A peak brighter than 1.0 clips all three channels at the
    // centre while the wings keep the hue — which is what a bright star looks like. It is a
    // consequence of the physics here, not an effect added on top.
    return vec4<f32>(in.color.rgb * intensity, 0.0);
}
