//! `vd-client-render` (Tier-B, P1.5 Slice 3) — the renderer glue around the pure
//! `vd-client-harness` + the renderer-free `vd-client` core.
//!
//! Skeleton until Slice-3 T4: the Bevy app (window + headless offscreen modes), the
//! billboard/egui draw, the `ImageCopyDriver` readback node (recipe validated in
//! `docs/design/spikes/bevy-readback/`), and the winit-input → `InputAction`-mailbox
//! bridge land here. ALL branching logic stays in `vd-client-harness` (Tier-A, 100%);
//! this crate is the wgpu/winit/egui/Bevy shell, proven by the HR6 visual harness +
//! `G-RENDER-SMOKE`, not by `llvm-cov` (it is intentionally absent from the Tier-A set).
