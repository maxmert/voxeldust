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

use vd_client_harness::camera::{CaptureCamera, fit_camera_to_bounds, framing_bounds};
use vd_core::glam::DVec3;
use vd_devproto::DevState;

/// Reconstruct the client's ACTUAL capture camera at the sampled state: the reported drawn boxes'
/// live centres + STREAMED extents → `framing_bounds` → `fit_camera_to_bounds` — the identical math
/// `frame_scene_camera` refits per frame, fed the identical inputs (the drawn scene). Residual
/// skew is sub-frame (the state sample vs the readback frame), bounded by one tick of orbital
/// motion against the scene radius.
///
/// ★ S5 (D-LOOK-3): the framing unions the SELF-AUTHORED OUTLINES (`body_kind == "look"`) — a
/// point of light states no outline, so there is nothing of it to frame — falling back to every
/// drawn centre when the picture holds no outline at all. Both sides call the one Tier-A
/// `framing_bounds`, so the reconstruction cannot drift from the renderer's own refit.
///
/// # Panics
/// On an empty drawn scene or a degenerate viewport — each a broken gate precondition, never a
/// soft skip.
#[must_use]
pub fn live_scene_camera(state: &DevState, width: usize, height: usize) -> CaptureCamera {
    let items: Vec<(DVec3, DVec3, bool)> = state
        .realm_boxes
        .iter()
        .map(|b| {
            (
                DVec3::from_array(b.center),
                DVec3::splat(b.extent_m),
                b.body_kind == "look",
            )
        })
        .collect();
    let (center, radius) = framing_bounds(&items)
        .expect("the client reports a non-empty drawn scene (realm_boxes) to frame");
    fit_camera_to_bounds(center, radius, width, height)
        .expect("the capture viewport is non-degenerate")
}
