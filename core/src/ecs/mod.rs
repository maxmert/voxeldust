//! Bevy ECS types shared between server shards and client.
//!
//! This module defines the component, resource, and event types that form the
//! entity-component architecture for Voxeldust. Game objects (ships, players)
//! are entities composed of these components. Systems query and mutate them.
//!
//! # Architecture
//!
//! - **Components**: per-entity state, split by access pattern for parallel queries.
//! - **Resources**: world-level singletons (physics context, planet positions, indices).
//! - **Events**: cross-system communication within a single tick (bridge from async networking).

pub mod components;
pub mod events;
pub mod resources;

pub use components::*;
pub use events::*;
pub use resources::*;
