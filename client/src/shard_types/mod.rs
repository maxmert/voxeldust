//! Concrete `ShardTypePlugin` implementations — one module per shard-type.
//!
//! Every plugin is a Bevy `Plugin` that (a) registers a trait object into
//! `ShardTypeRegistry` and (b) adds its own rendering / ingest / camera
//! systems via `render_contribution` (Phase 6+ — currently a no-op MVP
//! for every plugin; Phases 7/8/9 will add celestial bodies, sphere
//! projection, star catalog, etc.).
//!
//! Wire-level `shard_type` u8 constants (match
//! `voxydust/src/network.rs::is_scene_context` + server emitters):
//!   0 = Planet
//!   1 = System (scene-context)
//!   2 = Ship
//!   3 = Galaxy (scene-context)
//! Future additions (DEBRIS, STATION, …) append numerically.

pub mod galaxy;
pub mod planet;
pub mod ship;
pub mod starfield;
pub mod system;

/// Register every built-in shard-type plugin into the app. Future
/// shard-types are added here with one line apiece.
pub fn register_all(app: &mut bevy::prelude::App) {
    app.add_plugins((
        ship::ShipShardPlugin,
        system::SystemShardPlugin,
        planet::PlanetShardPlugin,
        galaxy::GalaxyShardPlugin,
        starfield::StarfieldPlugin,
    ));
}
