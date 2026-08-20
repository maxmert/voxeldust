//! THE seed universe generator (D-45(a); the placement arc S5) — the single source of truth for
//! realm→region geometry, now living where SL4 puts it: in the motion crate, beside the closed-form
//! celestial math it draws on. Closed-form `f(seed)`: every shard computes the IDENTICAL containment
//! forest at boot from the shared universe seed, so the geometry is REPLICATED BY CONSTRUCTION — no
//! shared mutable state, no inter-shard bytes (HR1).
//!
//! The split rule (SL5's guard, stated once): anything that READS THE SEED STREAM or MINTS A BODY
//! lives here; anything that only reads an already-built `&[RealmRegion]` stays in
//! `vd_core::worldgen` (topology utilities over this generator's output — `level_of`,
//! `coord_of_realm`, `ancestor_realms`, `pin_realm_of`, the neighbourhood scope). ONE generator; the
//! crossing path cannot name it (the `crate_isolation` SL4 law).
//!
//! The P3 forest models the mandate hierarchy **Universe ⊃ Galaxy ⊃ StarSystem ⊃ Planet** (walk scale:
//! Station + Area as first-class hand-placed realms — task #133); the visual/demand presets generate
//! the compressed-real Kepler systems every shipped shard boots (SL5: THE world, one of it).
//!
//! # ★ THE DISCOVERY-PERMANENCE LAW (owner ruling 2026-08-18, standing)
//!
//! **The generator is APPEND-ONLY in its draw stream.** A seed stream is positional exactly like
//! the wire: every draw's MEANING is its position, so inserting or reordering a draw re-rolls every
//! draw after it — and with it every orbit, every star, every albedo a player has already
//! discovered. Therefore:
//!
//! - a NEW derived quantity draws AFTER every existing draw of its stream, never between two
//!   (the practised pattern: the star's photometric draw appended after the planet elements, the
//!   albedo pass appended after the star, the 3-D placement pair appended after the albedos);
//! - every draw's stream POSITION is documented at its site and guarded by a bit-exact pin, so a
//!   reorder fails a named test before it can re-roll a world;
//! - a draw is never deleted while anything downstream of its position survives — retiring one
//!   means retiring the whole suffix behind a stated world-numbers change.
//!
//! What a player has found stays found: the world may gain content forever, and never loses or
//! moves what the seed already said.
//!
//! THE MODULE TREE (one lane per file; each file states what it owns and what it does not):
//! `config` the one config home and its defaults · `scale` the real-scale derivation chain and the
//! mass cap · `body` a generated body and how it lowers to a region · `aoi` a region's interest band
//! · `generate` the draws that make a system · `plant` the player-built fixtures · `walk` the
//! walk-scale mandate forest · `guards` the boot fences · `visibility` the climb measurement and its
//! two fences · `census` the Earth-like sweep · `forest_query` the roster questions a shard asks.
//!
//! The whole surface is re-exported here, so `vd_physics::worldgen::X` resolves exactly as it did
//! when this was one file.

mod aoi;
mod body;
mod census;
mod config;
mod forest_query;
mod generate;
mod guards;
mod plant;
mod scale;
mod visibility;
mod walk;

#[cfg(test)]
mod tests;

pub(crate) use aoi::*;
pub use body::*;
pub use census::*;
pub use config::*;
pub use forest_query::*;
pub use generate::*;
pub use guards::*;
pub use plant::*;
pub use scale::*;
pub use visibility::*;
pub use walk::*;
