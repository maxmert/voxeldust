//! The deterministic capture VERDICTS for the Visual Crossing Playground (Slice V1) — the
//! anti-vacuity oracle every visual crossing scenario asserts on.
//!
//! Three verdicts, each a pure function (Tier-A, 100% coverable, zero GPU):
//!
//! 1. **Membership (world-space, the deterministic artifact).** [`entity_in_box`] /
//!    [`expected_box`] test the COMPOSITED `world_pos` against a realm's box, REUSING the
//!    proptested `vd-core` box math ([`Boundary::signed_distance`] `<= 0` = inside, uniform for
//!    a sphere or a box). A crossing landed in the right box iff [`expected_box`] flips from the
//!    source realm to the dest realm — pure math, never GPU-inferred.
//!
//! 2. **Anti-vacuity crossing (adversary H1), pure-renderer form (S6).** [`crossing_was_real`] proves
//!    the dest stream ACTUALLY CARRIED the entity — a delivered track exists — so a re-home that only
//!    re-labeled box B by its `FrameRef` while the dest stream carried nothing FAILS the gate instead
//!    of passing vacuously. (The old "two-holder source+dest sub overlap" is deleted by the
//!    pure-renderer collapse: the client keys tracks by `EntityId` and never learns the owning node,
//!    so the same avatar on both subs folds into ONE track — a delivered track is the anti-vacuity.)
//!
//! 3. **Pixel corroboration (adversary H2).** [`dot_pixels_within_box_region`] confirms the dot's
//!    non-clear PIXELS fall inside the box's projected screen rectangle — the ONE legitimate
//!    screen-space step (reusing the Tier-A [`CaptureCamera::project_point`]); the membership
//!    verdict stays world-space. This is what makes "PIXEL-visible dot-crossing-a-boundary" an
//!    asserted fact, not just "something drew".

use glam::DVec3;
use vd_client::realm_scene::{BoxShape, RealmScene};
use vd_client::view::DeliveredView;
use vd_core::EntityId;
use vd_core::geometry::Boundary;
use vd_core::pose::RealmId;

use crate::camera::{CaptureCamera, ScreenAabb, ScreenPos};

/// The center-relative `vd-core` [`Boundary`] a [`BoxShape`] tests against — the reuse bridge:
/// a `Sphere`→`Shell`, a `Box`→`Aabb`, so the membership verdict runs the SAME proptested
/// `signed_distance` the shard uses (no duplicate branch logic; the Obb orientation is dropped at
/// projection, so a box tests axis-aligned — see the `BoxShape` NOTE in `vd-client`).
fn boundary_of(shape: BoxShape) -> Boundary {
    match shape {
        BoxShape::Sphere { r } => Boundary::Shell { r },
        BoxShape::Box { half } => Boundary::Aabb { half },
    }
}

/// Is the world point `world_p` inside realm `realm`'s box? Point-in-box membership on the
/// COMPOSITED world position, REUSING [`Boundary::signed_distance`] (`<= 0` = inside/on-surface,
/// uniform for sphere and box). `world_p` and the box are compared in the SAME render space: the box
/// is flattened through [`RealmBox::draw_center`], the one reduction the renderer uses.
/// Slice 5: this used to subtract the box's RAW centre from an origin-reduced point — correct only
/// while the world sits at cell zero. `false` for a realm not in the scene.
#[must_use]
pub fn entity_in_box(scene: &RealmScene, world_p: DVec3, realm: RealmId) -> bool {
    match scene.get(realm) {
        Some(rbox) => {
            let rel = world_p - rbox.draw_center();
            boundary_of(rbox.shape).signed_distance(rel) <= 0.0
        }
        None => false,
    }
}

/// The INNERMOST realm whose box contains `world_p`, or `None` if no box contains it. Deterministic:
/// scans the scene in `RealmId`-`Ord` order and keeps the DEEPEST containing box (nesting: a player
/// inside ship-inside-station resolves to the player's ship, not the station), ties broken by the
/// smaller `RealmId` (the scan keeps the first at a given depth, and iteration is `RealmId`-ordered).
#[must_use]
pub fn expected_box(scene: &RealmScene, world_p: DVec3) -> Option<RealmId> {
    let mut best: Option<(u8, RealmId)> = None;
    for (realm, rbox) in scene.iter() {
        let rel = world_p - rbox.draw_center();
        if boundary_of(rbox.shape).signed_distance(rel) <= 0.0 {
            let deeper = match best {
                Some((best_depth, _)) => rbox.depth > best_depth,
                None => true,
            };
            if deeper {
                best = Some((rbox.depth, realm));
            }
        }
    }
    best.map(|(_, realm)| realm)
}

/// **Anti-vacuity (H1), pure-renderer form (S6):** was the crossing REAL — does `entity` have a
/// DELIVERED track at this observed frame? A pure-renderer client keys tracks by `EntityId` (it never
/// learns which node owns the entity), so the old "two-holder source+dest sub overlap" is deleted:
/// the same avatar arriving on both subs folds into ONE track. The surviving anti-vacuity is that the
/// dest stream ACTUALLY CARRIED the entity — a delivered track exists — so asserting the dot then
/// landed in the dest box is NOT vacuous (a re-home that only re-labeled the `FrameRef` without a
/// real delivered pose returns `false`). `true` iff a delivered track exists for `entity`.
#[must_use]
pub fn crossing_was_real(view: &DeliveredView, entity: EntityId) -> bool {
    !view.subs_holding(entity).is_empty()
}

/// **Pixel corroboration (H2):** do the dot's non-clear PIXELS fall inside the box's projected
/// screen rectangle? Both AABBs are computed deterministically by the caller from
/// [`CaptureCamera::project_point`] (the dot's world pos → `dot_screen_aabb`; the box's projected
/// extent → `box_screen_aabb`). `true` iff every non-clear pixel of `rgba` inside `dot_screen_aabb`
/// also lies inside `box_screen_aabb` AND at least one such dot pixel exists (an empty dot region is
/// NOT a pass — that would be vacuous). Origin top-left, `(y*w+x)*4` indexing (matches `assert.rs`).
#[must_use]
pub fn dot_pixels_within_box_region(
    rgba: &[u8],
    w: usize,
    h: usize,
    clear: [u8; 4],
    dot_screen_aabb: ScreenAabb,
    box_screen_aabb: ScreenAabb,
) -> bool {
    let (x0, y0, x1, y1) = clamp_aabb_to_image(dot_screen_aabb, w, h);
    let mut saw_dot_pixel = false;
    for py in y0..y1 {
        for px in x0..x1 {
            let idx = (py * w + px) * CHANNELS;
            // A pixel outside the buffer is skipped (an oversized/off-image AABB cannot panic).
            let Some(pixel) = rgba.get(idx..idx + CHANNELS) else {
                continue;
            };
            if pixel == clear.as_slice() {
                continue; // a clear pixel is not the dot
            }
            saw_dot_pixel = true;
            // A non-clear (dot) pixel must lie inside the box's projected rectangle.
            let here = ScreenPos {
                x: px as f64,
                y: py as f64,
            };
            if !box_screen_aabb.contains(here) {
                return false;
            }
        }
    }
    // At least one dot pixel must have been seen — an empty region is a vacuous "pass", rejected.
    saw_dot_pixel
}

/// **Dot presence, background-differential (the dot-SENSITIVE half of the pixel verdicts — batch
/// review):** does `dot_screen_aabb` hold at least one pixel whose color differs from EVERY color on
/// the surrounding ring (`dot_screen_aabb` inflated by `ring_px`, minus its interior)? The ring
/// samples the LOCAL background — the translucent shell's uniform disc, or empty space — so a pixel
/// unlike all of it can only be the dot (no color contract needed: the dot's exact tonemapped value
/// is never assumed, only that it differs from what surrounds it). `false` when the dot region holds
/// no scannable pixel, or when everything in it matches the surround — a dot that did not draw FAILS
/// this, where the rect-containment verdict alone was satisfied by the background (its non-clear
/// pixels are anything at all, so a shell disc under the rect passed it dot or no dot).
///
/// DETERMINISTIC and threshold-free: exact byte equality against the ring's color SET, so a uniform
/// background of any color works and an MSAA-blended dot edge already differs. Callers must place
/// the ring where the background is locally uniform (deep inside one shell disc, or empty space) —
/// the render smokes assert that placement geometrically before asking this.
#[must_use]
pub fn dot_pixels_distinct_from_surround(
    rgba: &[u8],
    w: usize,
    h: usize,
    dot_screen_aabb: ScreenAabb,
    ring_px: f64,
) -> bool {
    let inner = clamp_aabb_to_image(dot_screen_aabb, w, h);
    let outer = clamp_aabb_to_image(
        ScreenAabb {
            min: ScreenPos {
                x: dot_screen_aabb.min.x - ring_px,
                y: dot_screen_aabb.min.y - ring_px,
            },
            max: ScreenPos {
                x: dot_screen_aabb.max.x + ring_px,
                y: dot_screen_aabb.max.y + ring_px,
            },
        },
        w,
        h,
    );
    // The surround's color set: every pixel of the outer rect that is NOT inside the inner rect.
    let mut surround: std::collections::BTreeSet<[u8; 4]> = std::collections::BTreeSet::new();
    let (ox0, oy0, ox1, oy1) = outer;
    let (ix0, iy0, ix1, iy1) = inner;
    for py in oy0..oy1 {
        for px in ox0..ox1 {
            let in_inner = (px >= ix0) & (px < ix1) & (py >= iy0) & (py < iy1);
            if in_inner {
                continue;
            }
            let idx = (py * w + px) * CHANNELS;
            let Some(pixel) = rgba.get(idx..idx + CHANNELS) else {
                continue; // an off-buffer pixel is skipped (an oversized AABB cannot panic)
            };
            surround.insert([pixel[0], pixel[1], pixel[2], pixel[3]]);
        }
    }
    // The dot region: any pixel unlike EVERYTHING on the ring is the dot.
    for py in iy0..iy1 {
        for px in ix0..ix1 {
            let idx = (py * w + px) * CHANNELS;
            let Some(pixel) = rgba.get(idx..idx + CHANNELS) else {
                continue;
            };
            if !surround.contains(&[pixel[0], pixel[1], pixel[2], pixel[3]]) {
                return true;
            }
        }
    }
    false
}

/// RGBA8: 4 bytes per pixel (mirrors `assert.rs`).
const CHANNELS: usize = 4;

/// Clamp a screen AABB to `[0,w) × [0,h)` integer pixel bounds as a half-open `(x0,y0,x1,y1)`
/// scan range (empty when the AABB is fully off-image or the image is empty). A monomorphic helper
/// so the clamp branches are covered here, off the pixel-scan body.
fn clamp_aabb_to_image(aabb: ScreenAabb, w: usize, h: usize) -> (usize, usize, usize, usize) {
    if w == 0 || h == 0 {
        return (0, 0, 0, 0);
    }
    // floor the min, ceil the max, then clamp into the image; a fully-negative or fully-past AABB
    // yields an empty range (x0 >= x1 or y0 >= y1), so the scan does nothing.
    let x0 = aabb.min.x.floor().clamp(0.0, w as f64) as usize;
    let y0 = aabb.min.y.floor().clamp(0.0, h as f64) as usize;
    let x1 = (aabb.max.x.ceil().clamp(0.0, w as f64) as usize).max(x0);
    let y1 = (aabb.max.y.ceil().clamp(0.0, h as f64) as usize).max(y0);
    (x0, y0, x1, y1)
}

/// The projected screen AABB of a world-space point sphere of radius `world_radius`, as the
/// bounding box of the center's projection expanded by the projection of an offset along the
/// camera's right axis — a convenience the pixel-verdict caller uses to build a box's or a dot's
/// screen rectangle deterministically. `None` if the center does not project (behind camera /
/// degenerate viewport).
#[must_use]
pub fn projected_point_aabb(
    camera: &CaptureCamera,
    world_center: DVec3,
    world_radius: f64,
) -> Option<ScreenAabb> {
    let center = camera.project_point(world_center)?;
    // Project a point offset by `world_radius` along the camera right axis to size the rectangle in
    // pixels (a conservative, deterministic screen radius). The right axis is view · +X. The offset
    // is PERPENDICULAR to the view forward, so it never changes the point's view-`w`: the edge
    // projects iff the center did (already gated above). We therefore fall back to `center` (a zero
    // screen radius = a degenerate point AABB) rather than carry a second, unreachable `?` early-out.
    let forward = (camera.target - camera.eye).normalize_or_zero();
    let right = forward.cross(camera.up).normalize_or_zero();
    let edge = camera
        .project_point(world_center + right * world_radius)
        .unwrap_or(center);
    let screen_radius = ((edge.x - center.x).powi(2) + (edge.y - center.y).powi(2)).sqrt();
    Some(ScreenAabb {
        min: ScreenPos {
            x: center.x - screen_radius,
            y: center.y - screen_radius,
        },
        max: ScreenPos {
            x: center.x + screen_radius,
            y: center.y + screen_radius,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use vd_client::interp::stated_tier;
    use vd_client::realm_scene::{
        BOX_ALPHA, MAX_NEST_DEPTH, RealmBox, SceneError, to_render_prims,
    };
    use vd_core::entity_kind::EntityKind;
    use vd_core::geometry::Boundary;
    use vd_core::pose::{FrameRef, LatticePos, StampedPose};
    use vd_core::{TickId, UniverseTick};
    use vd_wire::channels::SceneRow;
    use vd_wire::channels::SubId;
    use vd_wire::channels::{EntitySnap, SnapshotDatagram};

    fn ent(seq: u64) -> EntityId {
        EntityId::pack(EntityKind::Player, 1, seq, seq as u32)
    }

    /// One composed row with a self-authored look at `center` (post-flag-day fixture: the scene
    /// is built from SceneRows alone — the boundary/boot types never reach the client any more).
    fn look_row(
        realm: RealmId,
        parent: Option<RealmId>,
        center: DVec3,
        outline: Boundary,
    ) -> SceneRow {
        SceneRow {
            realm,
            parent,
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                center,
                UniverseTick(1),
            ),
            bag: vd_core::look::look_bag(&outline),
        }
    }

    /// A scene with one Shell realm (System 7, r=100 at origin) and one Aabb realm
    /// (Station 9, half=(50,50,50) offset at x=500) — both inside the renderable ceiling
    /// (post-flag-day the row's LOOK is skipped past `MAX_RENDERABLE_EXTENT_M`, exactly like the
    /// live composed level's ambient shells).
    fn two_realm_scene() -> RealmScene {
        RealmScene::from_scene_rows(&[
            look_row(
                RealmId::System(7),
                None,
                DVec3::ZERO,
                Boundary::Shell { r: 100.0 },
            ),
            look_row(
                RealmId::Station(9),
                None,
                DVec3::new(500.0, 0.0, 0.0),
                Boundary::Aabb {
                    half: DVec3::new(50.0, 50.0, 50.0),
                },
            ),
        ])
        .expect("scene")
    }

    #[test]
    fn entity_in_box_reuses_core_geometry_for_sphere_and_box() {
        let scene = two_realm_scene();
        // Inside the sphere (origin, r=100).
        assert!(entity_in_box(
            &scene,
            DVec3::new(50.0, 0.0, 0.0),
            RealmId::System(7)
        ));
        // On the sphere surface (== r) is inside (<= 0).
        assert!(entity_in_box(
            &scene,
            DVec3::new(100.0, 0.0, 0.0),
            RealmId::System(7)
        ));
        // Just outside the sphere.
        assert!(!entity_in_box(
            &scene,
            DVec3::new(100.1, 0.0, 0.0),
            RealmId::System(7)
        ));
        // Inside the box (center 500, half 50).
        assert!(entity_in_box(
            &scene,
            DVec3::new(500.0, 0.0, 0.0),
            RealmId::Station(9)
        ));
        assert!(entity_in_box(
            &scene,
            DVec3::new(550.0, 0.0, 0.0),
            RealmId::Station(9)
        ));
        assert!(!entity_in_box(
            &scene,
            DVec3::new(551.0, 0.0, 0.0),
            RealmId::Station(9)
        ));
        // A realm not in the scene → false (the None arm).
        assert!(!entity_in_box(&scene, DVec3::ZERO, RealmId::Planet(1)));
    }

    #[test]
    fn expected_box_flips_from_source_to_dest_across_a_crossing() {
        let scene = two_realm_scene();
        // At the origin the dot is in the System sphere.
        assert_eq!(
            expected_box(&scene, DVec3::new(0.0, 0.0, 0.0)),
            Some(RealmId::System(7))
        );
        // At x=500 it is in the Station box (the crossing destination).
        assert_eq!(
            expected_box(&scene, DVec3::new(500.0, 0.0, 0.0)),
            Some(RealmId::Station(9))
        );
        // Between the two, inside neither → None.
        assert_eq!(expected_box(&scene, DVec3::new(300.0, 0.0, 0.0)), None);
    }

    #[test]
    fn expected_box_picks_the_deepest_containing_realm_when_nested() {
        // A child box (depth 1) nested inside a parent (depth 0), both containing the point →
        // the deeper (child) wins (nesting semantics).
        let parent = RealmId::Station(1);
        let child = RealmId::System(2);
        let scene = RealmScene::from_scene_rows(&[
            look_row(
                parent,
                None,
                DVec3::ZERO,
                Boundary::Aabb {
                    half: DVec3::splat(100.0),
                },
            ),
            look_row(
                child,
                Some(parent),
                DVec3::ZERO,
                Boundary::Aabb {
                    half: DVec3::splat(10.0),
                },
            ),
        ])
        .expect("scene");
        // A point inside BOTH resolves to the child (depth 1 > depth 0).
        assert_eq!(expected_box(&scene, DVec3::new(1.0, 1.0, 1.0)), Some(child));
        // A point inside only the parent resolves to the parent.
        assert_eq!(
            expected_box(&scene, DVec3::new(50.0, 0.0, 0.0)),
            Some(parent)
        );
        // Outside both → None.
        assert_eq!(expected_box(&scene, DVec3::new(500.0, 0.0, 0.0)), None);
    }

    #[test]
    fn expected_box_of_an_empty_scene_is_none() {
        let scene = RealmScene::default();
        assert_eq!(expected_box(&scene, DVec3::ZERO), None);
    }

    fn snap_on(sub: SubId, frame_id: u64, tick: u64, entity: EntityId, x: f64) -> SnapshotDatagram {
        SnapshotDatagram {
            sub,
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            entities: vec![EntitySnap {
                entity,
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 1 },
                    DVec3::new(x, 0.0, 0.0),
                    UniverseTick(tick),
                ),
            }],
        }
    }

    #[test]
    fn crossing_was_real_requires_a_delivered_track() {
        // Pure-renderer anti-vacuity (S6): the crossing is real iff a DELIVERED track exists.
        let mut view = DeliveredView::default();
        let dot = ent(1);
        // No track yet → not real (a FrameRef-only re-label would be vacuous).
        assert!(!crossing_was_real(&view, dot));
        // A delivered pose on ANY held sub → real (the dest stream carried the entity).
        view.on_snapshot(
            &BTreeSet::from([SubId(1)]),
            snap_on(SubId(1), 1, 10, dot, 50.0),
        );
        assert!(crossing_was_real(&view, dot));
        // An entity with no delivered track at all → not real.
        assert!(!crossing_was_real(&view, ent(99)));
    }

    // ---- pixel corroboration ---------------------------------------------------

    const CLEAR: [u8; 4] = [10, 20, 30, 255];
    const DOT: [u8; 4] = [255, 255, 0, 255];

    fn solid(w: usize, h: usize, fill: [u8; 4]) -> Vec<u8> {
        fill.iter()
            .copied()
            .cycle()
            .take(w * h * CHANNELS)
            .collect()
    }
    fn set_px(buf: &mut [u8], w: usize, x: usize, y: usize, px: [u8; 4]) {
        let i = (y * w + x) * CHANNELS;
        buf[i..i + CHANNELS].copy_from_slice(&px);
    }

    #[test]
    fn dot_pixels_inside_the_box_region_pass_outside_fail() {
        let (w, h) = (8usize, 8usize);
        let box_region = ScreenAabb {
            min: ScreenPos { x: 2.0, y: 2.0 },
            max: ScreenPos { x: 5.0, y: 5.0 },
        };
        let dot_region = ScreenAabb {
            min: ScreenPos { x: 3.0, y: 3.0 },
            max: ScreenPos { x: 4.0, y: 4.0 },
        };
        // A dot pixel inside both regions AND inside the box → pass.
        let mut buf = solid(w, h, CLEAR);
        set_px(&mut buf, w, 3, 3, DOT);
        assert!(dot_pixels_within_box_region(
            &buf, w, h, CLEAR, dot_region, box_region
        ));
        // A dot pixel that is SCANNED but lies OUTSIDE the box region → fail via the `!contains`
        // return-false arm. The tight box is (2,2)-(4,4); the dot region (3,3)-(6,6) scans pixel
        // (5,5), which is outside that box rectangle, so the first non-clear pixel returns false.
        let tight_box = ScreenAabb {
            min: ScreenPos { x: 2.0, y: 2.0 },
            max: ScreenPos { x: 4.0, y: 4.0 },
        };
        let scan_dot = ScreenAabb {
            min: ScreenPos { x: 3.0, y: 3.0 },
            max: ScreenPos { x: 6.0, y: 6.0 },
        };
        let mut buf2 = solid(w, h, CLEAR);
        set_px(&mut buf2, w, 5, 5, DOT);
        assert!(!dot_pixels_within_box_region(
            &buf2, w, h, CLEAR, scan_dot, tight_box
        ));
    }

    #[test]
    fn a_pixel_index_past_a_truncated_buffer_is_skipped_not_a_panic() {
        // The `rgba.get(idx..idx+4)` None arm: `w`/`h` claim a full image but the buffer is
        // truncated, so a valid-looking index falls past the slice → that pixel is skipped (no
        // panic, no false pass). Only the surviving in-bounds clear pixels are seen → no dot → false.
        let (w, h) = (4usize, 4usize);
        // Full image is 4*4*4 = 64 bytes; truncate to two rows so the later scan indices are past it.
        let buf = solid(w, h, CLEAR)[..8 * CHANNELS].to_vec();
        let region = ScreenAabb {
            min: ScreenPos { x: 0.0, y: 0.0 },
            max: ScreenPos { x: 4.0, y: 4.0 },
        };
        assert!(!dot_pixels_within_box_region(
            &buf, w, h, CLEAR, region, region
        ));
    }

    #[test]
    fn an_empty_dot_region_is_not_a_vacuous_pass() {
        // No non-clear pixel in the dot region → false (the anti-vacuity clause).
        let (w, h) = (8usize, 8usize);
        let buf = solid(w, h, CLEAR); // all clear
        let dot_region = ScreenAabb {
            min: ScreenPos { x: 3.0, y: 3.0 },
            max: ScreenPos { x: 4.0, y: 4.0 },
        };
        let box_region = ScreenAabb {
            min: ScreenPos { x: 0.0, y: 0.0 },
            max: ScreenPos { x: 7.0, y: 7.0 },
        };
        assert!(!dot_pixels_within_box_region(
            &buf, w, h, CLEAR, dot_region, box_region
        ));
    }

    #[test]
    fn an_off_image_or_empty_buffer_dot_region_is_not_a_pass() {
        let box_region = ScreenAabb {
            min: ScreenPos { x: 0.0, y: 0.0 },
            max: ScreenPos { x: 100.0, y: 100.0 },
        };
        // A zero-size image → empty scan range → no dot pixel → false.
        assert!(!dot_pixels_within_box_region(
            &[],
            0,
            0,
            CLEAR,
            box_region,
            box_region
        ));
        // A zero-HEIGHT (but non-zero width) image → empty scan → false. Exercises the `h == 0`
        // operand of the `w == 0 || h == 0` guard (the `w == 0` short-circuit alone never reaches it).
        assert!(!dot_pixels_within_box_region(
            &[],
            4,
            0,
            CLEAR,
            box_region,
            box_region
        ));
        // A dot region entirely PAST the image → clamped to empty → false.
        let (w, h) = (4usize, 4usize);
        let buf = solid(w, h, CLEAR);
        let off = ScreenAabb {
            min: ScreenPos { x: 10.0, y: 10.0 },
            max: ScreenPos { x: 20.0, y: 20.0 },
        };
        assert!(!dot_pixels_within_box_region(
            &buf, w, h, CLEAR, off, box_region
        ));
        // A dot region entirely NEGATIVE (before the image) → clamped to empty → false.
        let neg = ScreenAabb {
            min: ScreenPos { x: -20.0, y: -20.0 },
            max: ScreenPos { x: -10.0, y: -10.0 },
        };
        assert!(!dot_pixels_within_box_region(
            &buf, w, h, CLEAR, neg, box_region
        ));
    }

    #[test]
    fn dot_region_larger_than_the_image_is_clamped_and_still_evaluated() {
        // An oversized dot AABB is clamped to the image (cannot panic) and still finds the dot.
        let (w, h) = (4usize, 4usize);
        let mut buf = solid(w, h, CLEAR);
        set_px(&mut buf, w, 1, 1, DOT);
        let big = ScreenAabb {
            min: ScreenPos { x: -5.0, y: -5.0 },
            max: ScreenPos { x: 100.0, y: 100.0 },
        };
        let box_region = ScreenAabb {
            min: ScreenPos { x: 0.0, y: 0.0 },
            max: ScreenPos { x: 3.0, y: 3.0 },
        };
        assert!(dot_pixels_within_box_region(
            &buf, w, h, CLEAR, big, box_region
        ));
    }

    #[test]
    fn a_dot_unlike_its_surround_is_present_a_matching_or_absent_one_is_not() {
        let (w, h) = (10usize, 10usize);
        let dot_region = ScreenAabb {
            min: ScreenPos { x: 4.0, y: 4.0 },
            max: ScreenPos { x: 6.0, y: 6.0 },
        };
        // PRESENT: a pixel inside the dot rect unlike everything on the ring (uniform background).
        let mut buf = solid(w, h, CLEAR);
        set_px(&mut buf, w, 4, 5, DOT);
        assert!(dot_pixels_distinct_from_surround(
            &buf, w, h, dot_region, 2.0
        ));
        // ABSENT: an all-background frame — the dot rect holds nothing the ring lacks.
        let flat = solid(w, h, CLEAR);
        assert!(!dot_pixels_distinct_from_surround(
            &flat, w, h, dot_region, 2.0
        ));
        // CAMOUFLAGED: the candidate's color ALSO appears on the ring (a second background tone),
        // so it proves nothing — the verdict refuses it.
        let mut two_tone = solid(w, h, CLEAR);
        set_px(&mut two_tone, w, 3, 3, DOT); // on the ring
        set_px(&mut two_tone, w, 5, 5, DOT); // in the dot rect, same color
        assert!(!dot_pixels_distinct_from_surround(
            &two_tone, w, h, dot_region, 2.0
        ));
        // OFF-IMAGE dot rect: clamped to empty — never a vacuous pass.
        let off = ScreenAabb {
            min: ScreenPos { x: 50.0, y: 50.0 },
            max: ScreenPos { x: 60.0, y: 60.0 },
        };
        assert!(!dot_pixels_distinct_from_surround(&buf, w, h, off, 2.0));
        // A CORNER rect: the ring clamps at the image edge and the verdict still works.
        let corner = ScreenAabb {
            min: ScreenPos { x: 0.0, y: 0.0 },
            max: ScreenPos { x: 2.0, y: 2.0 },
        };
        let mut corner_buf = solid(w, h, CLEAR);
        set_px(&mut corner_buf, w, 1, 1, DOT);
        assert!(dot_pixels_distinct_from_surround(
            &corner_buf,
            w,
            h,
            corner,
            2.0
        ));
    }

    #[test]
    fn a_truncated_buffer_is_skipped_in_both_scans_never_a_panic() {
        // The two `rgba.get` None arms (ring scan + dot scan): `w`/`h` claim a full image but the
        // buffer ends after one row, so later indices fall past the slice and are skipped.
        let (w, h) = (6usize, 6usize);
        let buf = solid(w, h, CLEAR)[..6 * CHANNELS].to_vec();
        let region = ScreenAabb {
            min: ScreenPos { x: 1.0, y: 1.0 },
            max: ScreenPos { x: 5.0, y: 5.0 },
        };
        assert!(!dot_pixels_distinct_from_surround(&buf, w, h, region, 1.0));
    }

    #[test]
    fn projected_point_aabb_brackets_a_projecting_center_and_rejects_behind_camera() {
        let camera = CaptureCamera {
            eye: DVec3::new(0.0, 0.0, 10.0),
            target: DVec3::ZERO,
            up: DVec3::Y,
            fov_y: std::f64::consts::FRAC_PI_2,
            width: 64,
            height: 48,
        };
        let aabb = projected_point_aabb(&camera, DVec3::ZERO, 1.0).expect("projects");
        // The origin projects to the viewport center (32,24); the AABB brackets it. Each bracket
        // is a separate `assert!` (HR5: no `&&` short-circuit branch in a helper).
        assert!(aabb.contains(ScreenPos { x: 32.0, y: 24.0 }));
        assert!(aabb.min.x < 32.0, "min.x brackets left of center");
        assert!(aabb.max.x > 32.0, "max.x brackets right of center");
        assert!(aabb.min.y < 24.0, "min.y brackets above center");
        assert!(aabb.max.y > 24.0, "max.y brackets below center");
        // A center behind the camera does not project → None.
        assert_eq!(
            projected_point_aabb(&camera, DVec3::new(0.0, 0.0, 100.0), 1.0),
            None
        );
    }

    #[test]
    fn to_render_prims_and_membership_agree_on_the_same_box() {
        // A sanity bridge: a RealmBox's membership (V1) is consistent with where V0 places it.
        let scene = {
            // Reconstruct a one-box scene from a composed row so entity_in_box has a realm key.
            RealmScene::from_scene_rows(&[look_row(
                RealmId::Station(1),
                None,
                DVec3::new(100.0, 0.0, 0.0),
                Boundary::Aabb {
                    half: DVec3::splat(10.0),
                },
            )])
            .expect("scene")
        };
        assert!(entity_in_box(
            &scene,
            DVec3::new(105.0, 0.0, 0.0),
            RealmId::Station(1)
        ));
        assert!(!entity_in_box(
            &scene,
            DVec3::new(120.0, 0.0, 0.0),
            RealmId::Station(1)
        ));
    }

    /// A unit `Aabb` composed row for `realm` with `parent` — a local helper for the scene-error
    /// paths below (the client-harness test binary exercises `from_scene_rows`' full branch
    /// surface too, HR5 rule (c)).
    fn aabb_b(realm: RealmId, parent: Option<RealmId>) -> SceneRow {
        look_row(
            realm,
            parent,
            DVec3::ZERO,
            Boundary::Aabb {
                half: DVec3::splat(1.0),
            },
        )
    }

    #[test]
    fn from_scene_rows_rejects_a_duplicate_realm_in_this_binary() {
        // Covers `from_scene_rows`' duplicate-reject arm from the client-harness binary too.
        let err = RealmScene::from_scene_rows(&[
            aabb_b(RealmId::Station(3), None),
            aabb_b(RealmId::Station(3), None),
        ])
        .expect_err("duplicate must reject");
        assert_eq!(err, SceneError::DuplicateRealm);
    }

    #[test]
    fn from_scene_rows_treats_an_out_of_scene_parent_as_a_root_in_this_binary() {
        // Covers `depth_of`'s out-of-scene-parent FALSE arm (parent not in the map) from this binary.
        let scene = RealmScene::from_scene_rows(&[aabb_b(
            RealmId::Station(4),
            Some(RealmId::System(99)), // System(99) absent from the set → the chain roots here
        )])
        .expect("projects");
        assert_eq!(scene.get(RealmId::Station(4)).expect("box").depth, 0);
    }

    #[test]
    fn from_scene_rows_bounds_a_parent_cycle_in_this_binary() {
        // Covers `depth_of`'s depth-ceiling TRUE arm (the cycle guard fires) from this binary.
        let a = RealmId::System(21);
        let b = RealmId::System(22);
        let err = RealmScene::from_scene_rows(&[aabb_b(a, Some(b)), aabb_b(b, Some(a))])
            .expect_err("cycle must reject");
        assert_eq!(err, SceneError::CycleOrDepthExceeded);
        // The ceiling const is the documented bound (referenced so the guard's intent is asserted).
        assert_eq!(MAX_NEST_DEPTH, 16);
    }

    #[test]
    fn to_render_prims_tessellates_both_a_sphere_and_a_box_in_this_binary() {
        // Covers the sphere pole-cap branches (`i != 0` / `i != STACKS-1`) and the box arm from the
        // client-harness binary — both tessellation paths must execute here, not only in vd-client.
        let sphere = RealmBox {
            shape: BoxShape::Sphere { r: 3.0 },
            body: vd_client::realm_scene::BodyKind::Look,
            luma: None,
            tier: stated_tier(FrameRef::SystemSpace { system_seed: 1 }),
            center: LatticePos::local(DVec3::ZERO),
            parent: None,
            depth: 0,
            color_rgba: [0.1, 0.2, 0.3, BOX_ALPHA],
        };
        let sphere_prims = to_render_prims(&sphere, DVec3::ZERO);
        assert_eq!(sphere_prims.len(), 1);
        // Every sphere vertex sits on the unit sphere and its normal equals its position.
        for v in &sphere_prims[0].vertices {
            let pos = DVec3::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            assert!((pos.length() - 1.0).abs() < 1e-6, "on the unit sphere");
            assert_eq!(v.pos, v.normal, "sphere normal == unit position");
        }
        let boxed = RealmBox {
            shape: BoxShape::Box {
                half: DVec3::splat(2.0),
            },
            body: vd_client::realm_scene::BodyKind::Look,
            luma: None,
            tier: stated_tier(FrameRef::SystemSpace { system_seed: 1 }),
            center: LatticePos::local(DVec3::ZERO),
            parent: None,
            depth: 0,
            color_rgba: [0.4, 0.5, 0.6, BOX_ALPHA],
        };
        let box_prims = to_render_prims(&boxed, DVec3::ZERO);
        assert_eq!(box_prims.len(), 1);
        // 6 faces × 2 tris × 3 verts = 36.
        assert_eq!(box_prims[0].vertices.len(), 36);
    }

    #[test]
    fn the_row_presence_gate_draws_looks_and_markers_and_skips_bagless_in_this_binary() {
        // The `from_boxes_json` loader this test used to cover is DELETED with the boot file
        // (Slice C1, D-LANE-6 🟩) — its rewritten subject is the ONE loader left: the composed
        // row's presence gate, exercised from the client-harness binary too (HR5(c)): a look row
        // draws as a body, a luma row as a zero-extent marker point, a bagless row is tracked but
        // never drawn.
        let marker_row = SceneRow {
            realm: RealmId::Planet(9),
            parent: Some(RealmId::System(7)),
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(30.0, 0.0, 0.0),
                UniverseTick(1),
            ),
            bag: vd_core::look::luma_bag(3, 0.25),
        };
        let bagless_row = SceneRow {
            realm: RealmId::Planet(10),
            parent: Some(RealmId::System(7)),
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(40.0, 0.0, 0.0),
                UniverseTick(1),
            ),
            bag: Vec::new(),
        };
        let scene = RealmScene::from_scene_rows(&[
            look_row(
                RealmId::System(7),
                None,
                DVec3::ZERO,
                Boundary::Shell { r: 40.0 },
            ),
            marker_row,
            bagless_row,
        ])
        .expect("the level projects");
        assert_eq!(scene.len(), 2, "the bagless row is tracked, never drawn");
        let marker = scene.get(RealmId::Planet(9)).expect("the marker point");
        assert_eq!(marker.body, vd_client::realm_scene::BodyKind::Marker);
        assert_eq!(marker.shape, BoxShape::Sphere { r: 0.0 });
        assert_eq!(marker.luma, Some((3, 0.25)));
        assert_eq!(
            scene.get(RealmId::System(7)).expect("the body").body,
            vd_client::realm_scene::BodyKind::Look,
        );
    }
}
