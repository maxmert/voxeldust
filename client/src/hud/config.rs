//! F-key config flow for HUD tiles.
//!
//! Today the config panel is a floating egui window. The long-term
//! target is: F summons a `HeldTablet` showing the target block's
//! config via the `ConfigPanelWidget` drawing onto the tile texture,
//! interactive through the in-game cursor. This module is the hook
//! point; the existing egui panel stays as the active path until the
//! tablet widget draw + focus-mode cursor land end-to-end.
//!
//! Wiring path (future):
//!   - Server sends `BlockConfigState` → `SpawnHeldTablet` fires with
//!     the config payload.
//!   - F or Escape → `DespawnHeldTablet`.
//!   - Apply button inside the `ConfigPanelWidget` → emit
//!     `ClientMsg::BlockConfigUpdate` via TCP + `DespawnHeldTablet`.

use bevy::prelude::*;

#[allow(dead_code)]
pub fn placeholder(_app: &mut App) {
    // Empty module today — hook point for the future tablet path.
    // Kept as a distinct file so the integration lands in one place
    // without sprawling into `tablet.rs` or `config_panel/`.
}
