//! Camera reconstruction for the GPU capture gates (Tier-B process glue). The client's offscreen
//! capture camera is `fit_camera_to_scene` over the LIVE overlaid scene, refit EVERY frame
//! (`frame_scene_camera` in `vd-client-render`) — so a gate that projects rectangles must
//! reconstruct THAT camera from the client's own reported drawn boxes, never fit its own over the
//! static boot file: the planets orbit, and an outer planet's apoapsis excursion past the star's
//! shell changes the union bounds and thus the camera (judge finding A1 — different cameras ⇒
//! wrong rectangles). What the client reports (`DevState.realm_boxes`) is each drawn box's label +
//! centre, not its size, so the extents come from THE world's own regions, joined by label.

use std::collections::BTreeMap;

use vd_client_harness::camera::{CaptureCamera, bounds_union, fit_camera_to_bounds};
use vd_core::geometry::{Boundary, RealmRegion};
use vd_core::glam::DVec3;
use vd_devproto::DevState;

/// A region shape's per-axis half-extent — the same reading the client's scene framing applies to
/// its lowered box shapes (a shell is `r` on every axis; a box is its `half`).
fn region_extent(shape: Boundary) -> DVec3 {
    match shape {
        Boundary::Shell { r } => DVec3::splat(r),
        Boundary::Aabb { half } | Boundary::Obb { half, .. } => half,
    }
}

/// The `{realm:?}`-label → half-extent map over THE world's regions — the join key is exactly the
/// label `DevState.realm_boxes` reports (`format!("{realm:?}")`), so a drawn box's size is read
/// from the world, never restated.
#[must_use]
pub fn extent_by_label(regions: &[RealmRegion]) -> BTreeMap<String, DVec3> {
    regions
        .iter()
        .map(|r| (format!("{:?}", r.realm), region_extent(r.shape)))
        .collect()
}

/// Reconstruct the client's ACTUAL capture camera at the sampled state: zip the reported drawn
/// boxes' live centres with THE world's extents by label → `bounds_union` → `fit_camera_to_bounds`
/// — the identical math `frame_scene_camera` refits per frame, fed the identical inputs (the drawn
/// scene). Residual skew is sub-frame (the state sample vs the readback frame), bounded by one tick
/// of orbital motion against a hundreds-of-metres scene radius.
///
/// # Panics
/// On a drawn box THE world does not name (the scene and the world diverged), an empty drawn scene,
/// or a degenerate viewport — each a broken gate precondition, never a soft skip.
#[must_use]
pub fn live_scene_camera(
    state: &DevState,
    extents: &BTreeMap<String, DVec3>,
    width: usize,
    height: usize,
) -> CaptureCamera {
    let items = state.realm_boxes.iter().map(|b| {
        let extent = extents.get(&b.realm).unwrap_or_else(|| {
            panic!(
                "the client draws a box ({}) that is not a region of THE world — the drawn scene \
                 and the emitted regions diverged",
                b.realm
            )
        });
        (DVec3::from_array(b.center), *extent)
    });
    let (center, radius) = bounds_union(items)
        .expect("the client reports a non-empty drawn scene (realm_boxes) to frame");
    fit_camera_to_bounds(center, radius, width, height)
        .expect("the capture viewport is non-degenerate")
}
