//! Per-class character asset registry.
//!
//! Holds the full lifecycle of every character class's GPU + animation
//! assets: handle creation → load polling → animation graph
//! construction → ready. The registry is keyed by
//! [`CharacterClass::id`], so a future race / cosmetic system can
//! register additional classes without touching the loader code.
//!
//! # State machine
//!
//! Each class transitions through four states once per session:
//!
//! ```text
//! Pending → Loading { handles } → BuildingGraph { handles }
//!                                       ↓
//!                                     Ready { ClassAssets }
//!                                       │
//!                                       └─→ on any asset failure: Failed
//! ```
//!
//! Phase B brings every class to `Ready` (or `Failed` if the user has
//! not dropped the `.glb` files yet). Phase C consumes
//! `ready(class_id)` to spawn skinned-mesh instances; Phase D consumes
//! the per-clip `AnimationNodeIndex` map to drive blends.

use std::collections::HashMap;

use bevy::animation::graph::AnimationNodeIndex;
use bevy::prelude::*;

use voxeldust_core::character::ClipLabel;

/// Per-class load state. Stored inside [`CharacterAssetRegistry`].
#[derive(Debug)]
pub enum AssetState {
    /// No load attempted yet — bridge state for the registry's
    /// initialization frame; a startup system replaces it with
    /// `Loading` on the next tick.
    Pending,
    /// Asset handles in flight via [`AssetServer`]. Each clip handle
    /// is paired with its [`ClipLabel`] so the loader can rebuild the
    /// label→node map once everything is resident.
    Loading {
        scene: Handle<Scene>,
        clips: Vec<(ClipLabel, Handle<AnimationClip>)>,
    },
    /// All handles loaded; AnimationGraph being constructed this tick.
    /// Transient — typically lives one frame.
    BuildingGraph {
        scene: Handle<Scene>,
        clips: Vec<(ClipLabel, Handle<AnimationClip>)>,
    },
    /// Steady-state. Every consumer should match on this variant.
    Ready(ClassAssets),
    /// Terminal failure — `reason` is logged once on entry and never
    /// retried (the registry treats `Failed` as final). Restart the
    /// client after fixing the asset on disk.
    Failed { reason: String },
}

impl AssetState {
    /// True when the state is in steady-state (`Ready`).
    #[inline]
    pub fn is_ready(&self) -> bool {
        matches!(self, AssetState::Ready(_))
    }

    /// True when no terminal state has been reached yet.
    #[inline]
    pub fn is_in_flight(&self) -> bool {
        matches!(
            self,
            AssetState::Pending
                | AssetState::Loading { .. }
                | AssetState::BuildingGraph { .. }
        )
    }
}

/// Steady-state asset bundle for one character class. Stored inside
/// [`AssetState::Ready`]. The contents are cheap to clone — every
/// field is a [`Handle`] or a primitive — so per-spawn instantiation
/// can pull out the scene + graph handle without locking the
/// registry.
#[derive(Debug, Clone)]
pub struct ClassAssets {
    /// Class metadata pointer — kept here so consumers don't need a
    /// separate lookup against [`voxeldust_core::character::class_by_id`].
    pub class: &'static voxeldust_core::character::CharacterClass,
    /// Skeleton + base mesh as a Bevy scene. Phase C spawns this via
    /// `SceneRoot(scene.clone())`; the SkinnedMesh + Skin components
    /// are auto-created by `bevy_gltf` as part of scene instantiation.
    pub scene: Handle<Scene>,
    /// Animation graph asset. Every character entity of this class
    /// references this single asset via `AnimationGraphHandle`, so all
    /// instances share the same node layout and weights are per-entity
    /// via `AnimationPlayer`.
    pub graph: Handle<AnimationGraph>,
    /// Dense map from `ClipLabel` (as `usize`) → graph node index.
    /// `None` slots indicate the class did not declare that clip —
    /// Phase D's animation driver gracefully skips missing clips.
    pub clip_nodes: [Option<AnimationNodeIndex>; ClipLabel::COUNT],
}

impl ClassAssets {
    /// Look up the graph node for a clip, or `None` if this class did
    /// not declare it.
    #[inline]
    pub fn node_for(&self, label: ClipLabel) -> Option<AnimationNodeIndex> {
        self.clip_nodes[label.index()]
    }
}

/// All character classes' asset state. Driven by the loader; consumed
/// by the spawn / animation driver systems.
#[derive(Resource, Default)]
pub struct CharacterAssetRegistry {
    /// Keyed by [`voxeldust_core::character::CharacterClass::id`].
    pub classes: HashMap<u16, AssetState>,
}

impl CharacterAssetRegistry {
    /// Convenience: typed view onto a class's `Ready` state. Returns
    /// `None` for any non-ready state (loading, failed, or unknown
    /// class id).
    #[inline]
    pub fn ready(&self, class_id: u16) -> Option<&ClassAssets> {
        match self.classes.get(&class_id) {
            Some(AssetState::Ready(a)) => Some(a),
            _ => None,
        }
    }

    /// True iff every registered class has reached `Ready`. Used by
    /// gating systems that wait for all classes to load before doing
    /// their first work (e.g. a "loading screen" overlay).
    pub fn all_ready(&self) -> bool {
        !self.classes.is_empty() && self.classes.values().all(AssetState::is_ready)
    }

    /// True iff any class has terminally failed.
    pub fn any_failed(&self) -> bool {
        self.classes
            .values()
            .any(|s| matches!(s, AssetState::Failed { .. }))
    }
}
