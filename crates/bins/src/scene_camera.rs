//! Camera reconstruction for the GPU capture gates (Tier-B process glue). The client's offscreen
//! capture camera is `fit_camera_to_scene` over the LIVE overlaid scene, refit EVERY frame
//! (`frame_scene_camera` in `vd-client-render`) — so a gate that projects rectangles must
//! reconstruct THAT camera from the client's own reported drawn boxes, never fit its own over a
//! file: the planets orbit, and an outer planet's apoapsis excursion past the star's shell changes
//! the union bounds and thus the camera (judge finding A1 — different cameras ⇒ wrong rectangles).
//!
//! Since the flag day (Slice C1, `docs/design/window_lane.md` §2.11) the extents are STREAMED:
//! `DevState.realm_boxes` carries each drawn box's label + centre + `extent_m` straight off the
//! composed row's look bag, so the reconstruction reads the client's own report whole — the
//! regions.json label-join died with the boot file (D-LANE-6 🟩).

use vd_client_harness::camera::{CaptureCamera, bounds_union, fit_camera_to_bounds};
use vd_core::glam::DVec3;
use vd_devproto::DevState;

/// Reconstruct the client's ACTUAL capture camera at the sampled state: the reported drawn boxes'
/// live centres + STREAMED extents → `bounds_union` → `fit_camera_to_bounds` — the identical math
/// `frame_scene_camera` refits per frame, fed the identical inputs (the drawn scene). Residual
/// skew is sub-frame (the state sample vs the readback frame), bounded by one tick of orbital
/// motion against a hundreds-of-metres scene radius. A MARKER box carries `extent_m == 0` — a
/// point contributes its centre to the union, exactly what the client's own framing does.
///
/// # Panics
/// On an empty drawn scene or a degenerate viewport — each a broken gate precondition, never a
/// soft skip.
#[must_use]
pub fn live_scene_camera(state: &DevState, width: usize, height: usize) -> CaptureCamera {
    let items = state
        .realm_boxes
        .iter()
        .map(|b| (DVec3::from_array(b.center), DVec3::splat(b.extent_m)));
    let (center, radius) = bounds_union(items)
        .expect("the client reports a non-empty drawn scene (realm_boxes) to frame");
    fit_camera_to_bounds(center, radius, width, height)
        .expect("the capture viewport is non-degenerate")
}
