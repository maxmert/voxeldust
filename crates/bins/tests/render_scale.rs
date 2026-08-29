//! **G-RENDER-SCALE** — THE RENDER PRECISION BUDGET, MEASURED ON THE WORLD (the S5 slice's own
//! measurement gate; `docs/design/DEFERRED.md` D-LOOK-3).
//!
//! Two numbers decide whether a picture spanning an avatar's own marker out to THE world's derived
//! star gap can be drawn at all, and both are measured here against the world's own geometry rather
//! than argued. (The span used to be quoted as `1e6` m to `2e15` m; the far end is
//! `cfg.stellar.galaxy_rim_r_m`, which the derived mass cap moved on 2026-08-20 — which is exactly
//! why the range ladder below is read and never written down.)
//!
//! 1. **THE DEPTH BUDGET.** Bevy builds a reverse-Z INFINITE perspective for a
//!    `PerspectiveProjection`, so the depth code is `near/z` and one `f32` code ulp is a CONSTANT
//!    fraction of the range at EVERY distance. This gate measures that fraction on the world's own
//!    range ladder and against the thinnest thing the world ever draws at each range — which is
//!    what says *reverse-Z alone spans the dynamic range and a logarithmic depth buys nothing*.
//! 2. **THE POSITION BUDGET.** The defect D-LOOK-3 records: an ABSOLUTE `f32` render position (and
//!    the `f32` `looking_at` subtraction built on it) loses whole kilometres — and, at the eye
//!    magnitudes THE world stands at, loses the camera's facing outright. The camera-relative
//!    flatten's residual is measured here in PIXELS beside it.
//!
//! No cluster, no GPU, no sleep: pure arithmetic over the booted world. It runs in seconds and is
//! the gate that would have caught the parked pixel gates' failure without a flight.
#![cfg(all(feature = "dev-control", feature = "render"))]

use vd_bins::DEV;
use vd_client_harness::camera::{
    DOT_MIN_APPARENT_RADIUS_PX, FIT_FOV_Y, depth_planes, eye_relative, look_rotation,
    one_pixel_world_m,
};
use vd_client_render::CAPTURE_H;
use vd_core::geometry::RealmRegion;
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;

/// THE REVERSE-Z DEPTH QUANTUM: with `depth = near/z`, one `f32` mantissa step of the code is one
/// `f32::EPSILON` of the code, which is the same RELATIVE step in `z` at every range. Named, not
/// transcribed: it IS `f32::EPSILON`.
const DEPTH_CODE_ULP: f64 = f32::EPSILON as f64;

fn world_config() -> vd_physics::worldgen::UniverseConfig {
    vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt)
}

fn regions() -> Vec<RealmRegion> {
    vd_physics::worldgen::realm_regions_for_config(DEV.universe_seed, &world_config())
}

/// The reverse-Z depth CODE a surface at `z` gets under a near plane of `near`, in `f32` — exactly
/// what the GPU stores.
fn depth_code(near: f64, z: f64) -> f32 {
    (near / z) as f32
}

/// ★ THE DEPTH BUDGET, measured on THE world.
///
/// THE LAW BEING MEASURED. A realm is DRAWN only while its own angular size clears the world's
/// visibility threshold, i.e. while `range ≤ look · spin_up_factor`. So at the FARTHEST range any
/// body is ever drawn as a body, its own front-to-back thickness is `2·look` over `look·factor` of
/// range — a CONSTANT `2/factor`, independent of the body, the seed and the scale. That constant is
/// the worst depth separation the picture ever has to resolve, and it is compared here against the
/// reverse-Z quantum.
#[test]
fn g_depth_budget_the_thinnest_thing_the_world_draws_out_resolves_the_reverse_z_quantum() {
    let cfg = world_config();
    let factor = cfg.interest.spin_up_factor;
    assert!(
        factor > 0.0,
        "the visibility factor must be live for the depth budget to mean anything",
    );
    // THE WORST DEPTH SEPARATION THE PICTURE EVER CARRIES, as a fraction of range.
    let worst_relative_thickness = 2.0 / factor;
    let margin = worst_relative_thickness / DEPTH_CODE_ULP;
    eprintln!(
        "[render-scale] visibility factor {factor:.6}; a body is drawn only while its own \
         thickness is >= 2/factor = {worst_relative_thickness:.6e} of its range. One reverse-Z f32 \
         depth code ulp is {DEPTH_CODE_ULP:.7e} of the range AT EVERY DISTANCE (that constancy is \
         why reverse-Z spans this world). MARGIN = {margin:.1}x.",
    );
    assert!(
        margin > 1.0,
        "DEPTH BUDGET BROKEN: the thinnest body the world draws separates by \
         {worst_relative_thickness:e} of its range, at or under the reverse-Z quantum \
         {DEPTH_CODE_ULP:e} — a second depth scheme (logarithmic depth) would be owed",
    );

    // AND THE CONCRETE WORST RANGE PAIR ON THE WORLD, both DERIVED: the outer world's own body at
    // the FARTHEST range it is ever drawn as a body (its own `look · spin_up_factor` — the law
    // stated above, so the pair is the law's own worst case rather than a range someone once
    // stood at), and the warp destination across THE world's own star gap. Both are resolved with
    // the DERIVED near plane the renderer installs, and the one-ulp depth step is printed in metres.
    // ★ RE-DERIVED 2026-08-21 (the gate-pass arc): the close-in range was the literal `1.8248e8` m,
    // an eye magnitude transcribed from a parked pixel gate's seed-0 run. Its thickness on the very
    // next line was already read live off the world, so the pair described two different worlds —
    // and the derived mass cap re-drew every star, moving the range again.
    let rs = regions();
    let home = vd_core::worldgen::default_home_realm(&rs).expect("THE world names a home realm");
    let outer_look = rs
        .iter()
        .filter(|r| r.parent == Some(home) && !matches!(r.realm, RealmId::Star(_)))
        .filter_map(|r| r.look.map(|l| l.finite_extent()))
        .fold(0.0_f64, f64::max);
    assert!(outer_look > 0.0, "THE world's home system draws its worlds");
    let star_gap = sibling_gap_m(&rs, home);
    let rows = f64::from(CAPTURE_H);
    for (name, range_m, thickness_m) in [
        (
            "the outer world's own body, at the farthest range it is still drawn as a body",
            outer_look * factor,
            2.0 * outer_look,
        ),
        (
            "the warp destination's point of light, across the star gap",
            star_gap,
            // A point of light is floored to the shared minimum apparent radius, so its drawn
            // thickness at range is that many pixels' worth of world — the camera's own answer.
            2.0 * DOT_MIN_APPARENT_RADIUS_PX * one_pixel_world_m(star_gap, FIT_FOV_Y, rows),
        ),
    ] {
        // The near plane the renderer installs for a picture whose nearest surface is the avatar's
        // own marker at arm's length and whose farthest is this subject.
        let planes = depth_planes(&[(range_m, thickness_m * 0.5)], FIT_FOV_Y, rows);
        let near_code = depth_code(planes.near, range_m - thickness_m * 0.5);
        let far_code = depth_code(planes.near, range_m + thickness_m * 0.5);
        let one_ulp_m = range_m * DEPTH_CODE_ULP;
        eprintln!(
            "[render-scale] {name}: range {range_m:.6e} m, drawn thickness {thickness_m:.6e} m, \
             derived near {:.6e} m ⇒ depth codes {near_code:e} vs {far_code:e}; ONE depth ulp is \
             {one_ulp_m:.6e} m of range ({:.1}x inside the thickness)",
            planes.near,
            thickness_m / one_ulp_m,
        );
        assert!(
            near_code > far_code,
            "{name}: the near and far faces must land on DISTINCT reverse-Z depth codes \
             ({near_code:e} vs {far_code:e}) or the body z-fights with itself",
        );
        assert!(
            thickness_m > one_ulp_m,
            "{name}: the drawn thickness {thickness_m:e} m is at or under one depth ulp \
             {one_ulp_m:e} m — z-fighting at this range is unavoidable",
        );
    }
}

/// ★ THE POSITION BUDGET — the defect, and the cure, in PIXELS on THE world's own ranges.
///
/// This is the measurement D-LOOK-3 is discharged on: at every eye magnitude THE world stands at,
/// an ABSOLUTE `f32` render position quantizes the picture, and the `f32` `looking_at` subtraction
/// built on it loses the camera's FACING outright. The render frame's residual is measured beside
/// it, in pixels of the shipped viewport.
#[test]
fn g_position_budget_the_render_frame_beats_the_absolute_f32_path_at_the_worlds_own_ranges() {
    let rs = regions();
    let home = vd_core::worldgen::default_home_realm(&rs).expect("THE world names a home realm");
    let home_bound = rs
        .iter()
        .find(|r| r.realm == home)
        .map(|r| r.shape.finite_extent())
        .expect("the home system is rostered");
    let star_gap = sibling_gap_m(&rs, home);
    let rows = f64::from(CAPTURE_H);
    let mut worst_render_px = 0.0_f64;
    let mut lost_facings = 0u32;
    for eye_mag in [home_bound, 2.0 * home_bound, star_gap] {
        // A general eye position (every component carries the magnitude) and a subject one metre
        // ahead of it — the exact shape `Transform::looking_at` is handed every frame.
        let eye = DVec3::splat(eye_mag / 3.0_f64.sqrt());
        let forward = DVec3::new(0.5, 0.5, -0.5).normalize();
        // (a) THE ABSOLUTE f32 PATH: the direction is LOST.
        let f32_direction = (eye + forward).as_vec3() - eye.as_vec3();
        lost_facings += u32::from(f32_direction == vd_core::glam::Vec3::ZERO);
        // (b) THE RENDER FRAME: the residual, in pixels at that range.
        let placed = eye_relative(eye + forward, eye);
        let residual_m = (placed - forward).length();
        let residual_px = residual_m / one_pixel_world_m(eye_mag, FIT_FOV_Y, rows);
        worst_render_px = worst_render_px.max(residual_px);
        // (c) And the f64 basis is unharmed at the same magnitude.
        let q = look_rotation(forward, DVec3::Y);
        let basis_err = (q * DVec3::Z + forward).length();
        eprintln!(
            "[render-scale] |eye| {eye_mag:.6e} m: f32 look-at direction {f32_direction:?} \
             (ZERO ⇒ the camera falls back to facing world −Z); render-frame residual \
             {residual_m:.6e} m = {residual_px:.3e} px; f64 basis error {basis_err:.3e}",
        );
        assert!(
            basis_err < 1e-12,
            "the f64 camera basis must still face the mark at |eye| = {eye_mag:e} m",
        );
    }
    assert_eq!(
        lost_facings, 3,
        "THE DEFECT MUST STILL BE REAL at every one of THE world's eye magnitudes — if an f32 \
         look-at direction survived here, this gate would be measuring nothing",
    );
    assert!(
        worst_render_px < 1.0,
        "the render frame's worst residual is {worst_render_px:e} px — the flatten must keep the \
         whole picture inside one pixel at every range THE world stands at",
    );
    eprintln!(
        "[render-scale] the render frame's WORST residual over THE world's eye ladder: \
         {worst_render_px:.3e} px",
    );
}

/// The distance from the home system to the FARTHEST star system the galaxy authors — the longest
/// range the picture ever has to hold, read off the placements the galaxy itself states (SL1).
fn sibling_gap_m(rs: &[RealmRegion], home: RealmId) -> f64 {
    let galaxy = rs
        .iter()
        .find(|r| r.realm == home)
        .and_then(|r| r.parent)
        .expect("the home system nests under the galaxy");
    let at = |realm: RealmId| {
        rs.iter()
            .find(|r| r.realm == realm)
            .map(|r| {
                // COMPILE-RESTORED against `ParentCentre` (S9), with the arithmetic UNCHANGED.
                //
                // ⚠ SUSPECT: `center` is stated in the PARENT's frame, while `r.frame.tier()` is this
                // region's OWN tier. Reading one with the other's ruler is the 2048x parent-frame trap
                // this tree has hit before. It is left exactly as it was rather than "corrected" here,
                // because changing a measurement inside a scale test is a decision that needs its own
                // analysis, not a drive-by while restoring the build. See the note in DEFERRED.md.
                r.center
                    .in_parents_frame()
                    .delta_m(vd_core::pose::LatticePos::ORIGIN, r.frame.tier())
            })
            .expect("the realm is rostered")
    };
    let here = at(home);
    rs.iter()
        .filter(|r| r.parent == Some(galaxy) && r.realm != home)
        .filter(|r| matches!(r.realm, RealmId::System(_)))
        .map(|r| (at(r.realm) - here).length())
        .fold(0.0_f64, f64::max)
}
