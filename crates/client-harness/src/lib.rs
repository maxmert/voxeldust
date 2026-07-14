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

pub mod assert;
pub mod camera;
pub mod capture;
pub mod input_map;
pub mod manifest;
pub mod nav;
pub mod verdict;
