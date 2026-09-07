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
    // ★ THE POINT-SOURCE LAW'S OWN NUMBERS (2026-08-29), every one read from Tier-A's `StarTuning` at
    // bind time. Nothing here is typed as a literal: the law lives in `vd_client::realm_scene`, is
    // unit-tested there, and this stage is its transliteration — the same discipline
    // `one_pixel_world_m` already follows.
    // √(flux gain): applied BEFORE the square so `(base·√gain/d)²` never underflows f32 (see the
    // Rust side). The law is the same `L/d²` up to the exposure; only the arithmetic order moved.
    flux_gain_sqrt: f32,
    response_exponent: f32,
    halo_sigma_px: f32,
    halo_weight: f32,
    core_sigma_px: f32,
    min_crop_px: f32,
    cull_level: f32,
};

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

    // ★ THE FLUX, FROM WHAT THE VERTEX ALREADY CARRIES (2026-08-29). `base_radius_m` is
    // `POINT_SOURCE_BASE_RADIUS_M · sqrt(L)`, so `(base/d)^2` IS `L/d^2` — the inverse-square law,
    // with no new attribute and no wire change.
    //
    // This REPLACES the shared 3-pixel apparent-size floor, which clamped every star to the same
    // size. MEASURED: at a typical neighbour distance that floor was worth 4.1e14 m against a
    // sun-like star's 0.5 m — it won by 8e14, for every star, always. Luminosity was drawn from the
    // seed, carried across the wire, and then discarded at the last step.
    // ★ SCALED BEFORE THE SQUARE (2026-09-07): `base/d` is ~1e-19 for a faint far star and its
    // square is below f32's smallest normal, which the GPU flushes to zero — the star vanished.
    // With √gain folded in first the term is ~1e-2..1e3 and the square is exact.
    let s = in.base_radius_m * star.flux_gain_sqrt / max(dist_m, 1.0);
    let flux = s * s;
    // ★ COMPRESSED, BECAUSE SIGHT IS. Linear flux made near stars into saturated white balls 19 px
    // across — a galaxy's flux range is a million to one and a screen's is 255. Eyes, film and star
    // charts are all logarithmic; stellar MAGNITUDE is exactly this compression, and a small power
    // is the same curve in closed form.
    let amplitude = pow(flux, star.response_exponent);

    // ★ CROP, NEVER SCALE — the idea from `godot-starlight` (Tiffany Bennett, MIT). One fixed profile
    // for every star, cut where its wings fade below what a screen can show:
    //     A·w·exp(-r^2 / 2σ^2) = cull   ⇒   r = σ·sqrt(2·ln(A·w/cull))
    // A bright star's quad is large because its faint outskirts stay visible further out, not because
    // its core is wider. That is what a star IS, and it is also why this costs less than today: the
    // faint majority draw SMALLER than the 3 pixels they all take now.
    let ratio = amplitude * star.halo_weight / star.cull_level;
    var crop_px = 0.0;
    if (ratio > 1.0) {
        crop_px = star.halo_sigma_px * sqrt(2.0 * log(ratio));
    }
    // NEVER SMALLER THAN THIS. A quad under a pixel misses the pixel centre and BLINKS as the camera
    // turns — the failure Stellarium and Celestia both name. A star fades out; it never flickers out.
    crop_px = max(crop_px, star.min_crop_px);

    // CULLED ON BRIGHTNESS, NEVER ON SIZE: collapse the quad so it rasterises nothing. A star leaves
    // the sky because it faded, which is continuous, and not because it got small, which pops.
    if (amplitude <= star.cull_level) {
        crop_px = 0.0;
    }

    // THE SPRITE AT ITS PLACED DEPTH. `crop_px` is a size in PIXELS, so it converts through one
    // pixel's world size AT THE DEPTH THE SPRITE IS DRAWN — not at the star's true distance, which
    // would balloon a far star to the size it would have had if it were close.
    let placed_radius_m = crop_px * one_pixel_world_m(star.sky_radius_m);

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
    // ★ A STAR IS A PEAK WITH WINGS, NOT A FILLED DISC (2026-08-29).
    //
    // This stage used to return `alpha = 1.0` across the whole interior with a one-pixel soft rim —
    // a flat plate of uniform colour, which is the definition of a dot. Together with the size floor
    // that clamped every star to 3 pixels, that is exactly what the owner saw: 233 220 identical
    // pieces of confetti.
    //
    // What you actually see when you look at a star is your own eye's blur — a bright narrow CORE
    // inside a wide faint HALO. Both widths are properties of the eye, so they are the same for every
    // star; only the amplitude differs. Two Gaussians are the standard model and are enough.
    //
    // `in.corner` runs -1..1 across the quad, and the quad was cropped to `crop_px` — so the radius in
    // PIXELS is the corner times that crop, which is what the profile's sigmas are measured in.
    let r_px = length(in.corner) * (in.crop_px);
    let core = exp(-0.5 * (r_px * r_px) / (star.core_sigma_px * star.core_sigma_px));
    let halo = exp(-0.5 * (r_px * r_px) / (star.halo_sigma_px * star.halo_sigma_px));
    let profile = core + star.halo_weight * halo;

    let intensity = in.amplitude * profile;
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
