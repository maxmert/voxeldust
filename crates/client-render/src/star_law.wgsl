// ★ THE STAR SPRITE'S LAW (S11; 2026-08-29) — ONE source for the star cloud's two draws: the sky's own
// material in the main pass (`star_sky.wgsl`) and, under the temporal path, the points pass after the
// reconstruction (`points.wgsl`; the design §7.7), which draws exactly today's sprites at the display size.
//
// ★ EVERY CONSTANT ARRIVES AS A UNIFORM. Nothing here is a literal from the size law. The law lives in
// Tier-A (`vd_client::realm_scene::star_draw`, from `StarTuning`), is unit-tested there, and this module is
// its transliteration.
//
// ★ THIS MODULE DOES NOT CHOOSE A BRIGHTNESS FALLOFF. Brightness is a live owner knob with an undecided curve
// (D-WINDOW-3). Size carries luminosity, through the square-root convention that Tier-A already owns.

#define_import_path vd_client_render::star_law

struct StarSkyUniform {
    // The apparent-size floor, in pixels, from DOT_MIN_APPARENT_RADIUS_PX.
    min_apparent_radius_px: f32,
    // tan(fov_y / 2) — half the vertical field of view.
    tan_half_fov: f32,
    // The viewport height in rows (the DISPLAY picture's: the law's sizes are display pixels, F24).
    viewport_h_px: f32,
    // WHERE THE SKY SITS IN DEPTH, in metres — just inside the camera's far plane. The star's true
    // DIRECTION is kept, and only its depth slot is fixed (see `star_sky.wgsl`).
    sky_radius_m: f32,
    // ★ THE POINT-SOURCE LAW'S OWN NUMBERS (2026-08-29), every one read from Tier-A's `StarTuning` at
    // bind time.
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

// THE WORLD SIZE OF ONE PIXEL at a view distance — a transliteration of Tier-A's `one_pixel_world_m`.
// A subject at or behind the eye has no forward extent, so the distance floors at zero.
fn one_pixel_world_m(star: StarSkyUniform, dist_m: f32) -> f32 {
    return 2.0 * max(dist_m, 0.0) * star.tan_half_fov / star.viewport_h_px;
}

// ★ THE AMPLITUDE of a star from what its vertex carries.
//
// `base_radius_m` is `POINT_SOURCE_BASE_RADIUS_M · sqrt(L)`, so `(base/d)^2` IS `L/d^2` — the inverse-square
// law, with no new attribute and no wire change. It REPLACED the shared 3-pixel apparent-size floor, which
// clamped every star to the same size (MEASURED: at a typical neighbour distance that floor won by 8e14,
// for every star, always).
// ★ SCALED BEFORE THE SQUARE (2026-09-07): `base/d` is ~1e-19 for a faint far star and its square is below
// f32's smallest normal, which the GPU flushes to zero — the star vanished. With √gain folded in first the
// term is ~1e-2..1e3 and the square is exact.
// ★ COMPRESSED, BECAUSE SIGHT IS. Linear flux made near stars into saturated white balls 19 px across — a
// galaxy's flux range is a million to one and a screen's is 255. Stellar MAGNITUDE is exactly this
// compression, and a small power is the same curve in closed form.
fn star_amplitude(star: StarSkyUniform, base_radius_m: f32, dist_m: f32) -> f32 {
    let s = base_radius_m * star.flux_gain_sqrt / max(dist_m, 1.0);
    let flux = s * s;
    return pow(flux, star.response_exponent);
}

// ★ CROP, NEVER SCALE — the idea from `godot-starlight` (Tiffany Bennett, MIT). One fixed profile for every
// star, cut where its wings fade below what a screen can show:
//     A·w·exp(-r^2 / 2σ^2) = cull   ⇒   r = σ·sqrt(2·ln(A·w/cull))
// A bright star's quad is large because its faint outskirts stay visible further out, not because its core
// is wider. NEVER SMALLER THAN the minimum crop: a quad under a pixel misses the pixel centre and BLINKS as
// the camera turns — the failure Stellarium and Celestia both name. CULLED ON BRIGHTNESS, NEVER ON SIZE: the
// quad collapses and rasterises nothing, so a star leaves the sky because it faded, which is continuous.
fn star_crop_px(star: StarSkyUniform, amplitude: f32) -> f32 {
    let ratio = amplitude * star.halo_weight / star.cull_level;
    var crop_px = 0.0;
    if (ratio > 1.0) {
        crop_px = star.halo_sigma_px * sqrt(2.0 * log(ratio));
    }
    crop_px = max(crop_px, star.min_crop_px);
    if (amplitude <= star.cull_level) {
        crop_px = 0.0;
    }
    return crop_px;
}

// ★ A STAR IS A PEAK WITH WINGS, NOT A FILLED DISC (2026-08-29): your own eye's blur — a bright narrow CORE
// inside a wide faint HALO, both widths the eye's, the same for every star; only the amplitude differs.
// `r_px` is the distance from the sprite's centre in PIXELS, the unit the profile's sigmas are stated in.
fn star_profile(star: StarSkyUniform, r_px: f32) -> f32 {
    let core = exp(-0.5 * (r_px * r_px) / (star.core_sigma_px * star.core_sigma_px));
    let halo = exp(-0.5 * (r_px * r_px) / (star.halo_sigma_px * star.halo_sigma_px));
    return core + star.halo_weight * halo;
}
