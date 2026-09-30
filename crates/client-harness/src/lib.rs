//! `vd-client-harness` (Tier-A, P1.5 Slice 3) — the PURE harness logic the renderer
//! glue (`vd-client-render`) and the dev-control listener stand on. ZERO renderer/GPU
//! deps; 100% region+branch coverage; every operational threshold is a named const.
//!
//! - [`assert`] — the HR6 visual-regression checks over a readback RGBA8/f32 buffer
//!   (`render_smoke` = the G-RENDER-SMOKE verdict).
//! - [`camera`] — the follow-own-dot first-person camera math (up-vector parameterized
//!   for planet-radial / ship-local later) + the `CaptureCamera` world→screen projection
//!   (the ONE legitimate screen-space step, for the pixel-corroboration verdict).
//! - [`verdict`] — the Visual Crossing Playground capture VERDICTS: world-space box
//!   membership (`entity_in_box`/`expected_box`), the anti-vacuity `crossing_was_real`,
//!   and the `dot_pixels_within_box_region` pixel corroboration.
//! - [`input_map`] — keyboard/mouse → `InputAction` (raw + opaque; the key→signal
//!   binding is SERVER-side, never here — slice_3 plan §10).
//! - [`nav`] — the walk-to / look-at closed-loop math (in the stub's local movement
//!   frame: `step = orient * (strafe, vert, -fwd) · speed·dt`).
//! - [`manifest`] — the `runs/` capture manifest serde (tick ↔ frame ↔ state).
//! - [`capture`] — the `--at-tick` capture-alignment (reuses the wait-until predicate).
//! - [`probe`] — the picture instrument's pure half (slice 8p): the probe codec, the stamp's
//!   formulas, the blob reader the gates measure a probe with.
//! - [`edge`] — the edge meter (G-EDGE, the temporal path's slice T0): the blended share and
//!   the blend levels of the horizon and limb pixels, read on a picture and its probe.
//! - [`flicker`] — the flicker meter's pure half (G-FLICKER, the temporal path's slice T0): each
//!   pixel's step from frame to frame at a still stand, by region, against the proposed targets.
//! - [`stars`] — the star census (G-STARS, the temporal path's slice T0): a candidate run's stars
//!   against today's picture, frame by frame (turned with the camera in a view that turns), their
//!   summed light, and the design's §12.4 verdict.
//! - [`ghost`] — the ghost rig (G-GHOST, the temporal path's slice T0): the trail behind a moving
//!   hull box and a dot, in display pixels, on three scan lines against the background.
//! - [`sharp`] — the sharpness meter G-SHARP (the temporal path, slice T0): the fine-detail energy
//!   of the ground's pixels, raw and normalised by the local contrast, per distance band.
//! - [`picture_census`] — the picture census (the speed step's slice T0, G-PICTURE-HOLES): the
//!   ground pixels of the probe that keep the sentinel clear colour in the picture, which are the
//!   holes of the picture camera's own depth test that the probe cannot see.
//! - [`motion_census`] — the motion census (the temporal path's slice T1, G-MOTION-COVERAGE): the
//!   captured motion target and depth, and the pixels that hold a depth yet keep the motion sentinel.

pub mod air_line;
pub mod assert;
pub mod camera;
pub mod capture;
pub mod edge;
pub mod flicker;
pub mod ghost;
pub mod input_map;
pub mod manifest;
pub mod motion_census;
pub mod nav;
pub mod picture_census;
pub mod pop;
pub mod probe;
pub mod sharp;
pub mod stars;
pub mod verdict;
