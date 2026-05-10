//! Character asset loading state machine.
//!
//! Two systems drive the [`super::assets::CharacterAssetRegistry`]:
//!
//! 1. [`enqueue_class_loads`] (Startup): for every class in
//!    [`voxeldust_core::character::ALL_CLASSES`], create the
//!    asset handles and stash them as `AssetState::Loading`.
//! 2. [`poll_class_loads`] (Update): per frame, check the load state
//!    of each `Loading` class; on full load build the
//!    `AnimationGraph`, transition to `Ready`. On any asset failure
//!    transition to `Failed` and log a clear, actionable error.
//!
//! No locks, no async tasks — Bevy's `AssetServer` is the async
//! infrastructure; we just poll its synchronous API each tick.
//!
//! # Why an explicit state machine
//!
//! Bevy supports `AssetEvent::LoadedWithDependencies`, but events
//! drop after one frame and we'd have to reconstruct partial state on
//! every system run. A simple per-class `AssetState` enum is cheaper
//! and self-evidently correct.

use bevy::animation::graph::{AnimationGraph, AnimationNodeIndex};
use bevy::asset::LoadState;
use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;

use voxeldust_core::character::{ClipLabel, ALL_CLASSES};

use super::assets::{AssetState, CharacterAssetRegistry, ClassAssets};

/// Startup system: stash a fresh `AssetState::Loading` for every
/// registered class. Idempotent on repeated startup-set runs (replaces
/// any existing entry for the class id) so a future hot-reload path
/// can drive it through `Startup` again.
pub(super) fn enqueue_class_loads(
    asset_server: Res<AssetServer>,
    mut registry: ResMut<CharacterAssetRegistry>,
) {
    for class in ALL_CLASSES {
        // Bevy's `bevy_gltf` plugin registers a loader that produces
        // `Gltf` from a raw `.glb`. Loading `Scene` or `AnimationClip`
        // directly requires the `#SceneN` / `#AnimationN` sub-asset
        // label so the loader knows which inner asset to materialize.
        // Mixamo character/animation exports each contain exactly one
        // scene + one animation track, so index 0 is always correct.
        let scene_handle: Handle<Scene> =
            asset_server.load(GltfAssetLabel::Scene(0).from_asset(class.asset_path));
        let clips: Vec<(ClipLabel, Handle<AnimationClip>)> = class
            .clips
            .iter()
            .map(|c| {
                (
                    c.label,
                    asset_server.load::<AnimationClip>(
                        GltfAssetLabel::Animation(0).from_asset(c.path),
                    ),
                )
            })
            .collect();

        info!(
            class = class.name,
            id = class.id,
            scene = class.asset_path,
            clip_count = clips.len(),
            "character asset: load enqueued"
        );

        registry.classes.insert(
            class.id,
            AssetState::Loading {
                scene: scene_handle,
                clips,
            },
        );
    }
}

/// Per-frame poll. Cheap when every class is `Ready` (early-out via
/// `is_in_flight()`).
pub(super) fn poll_class_loads(
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut registry: ResMut<CharacterAssetRegistry>,
) {
    // Fast path — nothing to do once every class has terminated.
    if registry
        .classes
        .values()
        .all(|s| !s.is_in_flight())
    {
        return;
    }

    // Resolve a fresh transition decision for every in-flight class
    // before mutating the map; avoids overlapping borrows on `registry`.
    let class_ids: Vec<u16> = registry
        .classes
        .iter()
        .filter(|(_, s)| s.is_in_flight())
        .map(|(id, _)| *id)
        .collect();

    for class_id in class_ids {
        let new_state = match registry.classes.get(&class_id) {
            Some(AssetState::Loading { scene, clips }) => {
                check_loading(&asset_server, &mut graphs, class_id, scene, clips)
            }
            Some(AssetState::BuildingGraph { scene, clips }) => {
                // Reached only if the previous tick saw `LoadState::Loaded`
                // for every handle but graph build was deferred. Build
                // now and finish.
                Some(build_ready(&mut graphs, class_id, scene.clone(), clips))
            }
            // `Pending`, `Ready`, `Failed` are no-ops here.
            _ => None,
        };

        if let Some(new_state) = new_state {
            registry.classes.insert(class_id, new_state);
        }
    }
}

/// Inspect the `Loading` state and decide the transition. Returns
/// `None` to keep the state unchanged (still loading).
fn check_loading(
    asset_server: &AssetServer,
    graphs: &mut Assets<AnimationGraph>,
    class_id: u16,
    scene: &Handle<Scene>,
    clips: &[(ClipLabel, Handle<AnimationClip>)],
) -> Option<AssetState> {
    let scene_state = asset_server.get_load_state(scene);

    let scene_failed = matches!(scene_state, Some(LoadState::Failed(_)));
    let scene_loaded = matches!(scene_state, Some(LoadState::Loaded));

    // Aggregate clip load state in one pass.
    let mut all_clips_loaded = true;
    let mut any_clip_failed = false;
    for (_, h) in clips {
        match asset_server.get_load_state(h) {
            Some(LoadState::Loaded) => {}
            Some(LoadState::Failed(_)) => {
                any_clip_failed = true;
                all_clips_loaded = false;
            }
            _ => {
                all_clips_loaded = false;
            }
        }
    }

    if scene_failed || any_clip_failed {
        // Diagnose precisely so the user knows what to fix.
        let class = voxeldust_core::character::class_by_id(class_id)
            .map(|c| c.name)
            .unwrap_or("?");
        let mut missing: Vec<&str> = Vec::new();
        if scene_failed {
            // We can't recover the path from the handle directly without
            // a registry; look up via class metadata.
            if let Some(c) = voxeldust_core::character::class_by_id(class_id) {
                missing.push(c.asset_path);
            }
        }
        for (label, h) in clips {
            if matches!(asset_server.get_load_state(h), Some(LoadState::Failed(_))) {
                if let Some(c) = voxeldust_core::character::class_by_id(class_id) {
                    if let Some(p) = c.clip_path(*label) {
                        missing.push(p);
                    }
                }
            }
        }
        let reason = format!(
            "missing or unreadable assets: {}",
            missing.join(", "),
        );
        error!(
            class,
            reason = %reason,
            "character asset: LOAD FAILED — drop the .glb files into client/assets/ and restart"
        );
        return Some(AssetState::Failed { reason });
    }

    if scene_loaded && all_clips_loaded {
        let new_state = build_ready(graphs, class_id, scene.clone(), clips);
        return Some(new_state);
    }

    // Still loading.
    None
}

/// Build the `AnimationGraph`, log the success line, and return the
/// `Ready` state. Called once per class.
fn build_ready(
    graphs: &mut Assets<AnimationGraph>,
    class_id: u16,
    scene: Handle<Scene>,
    clips: &[(ClipLabel, Handle<AnimationClip>)],
) -> AssetState {
    let class = match voxeldust_core::character::class_by_id(class_id) {
        Some(c) => c,
        None => {
            // Defensive: a class id ended up in the registry without
            // matching metadata. This is a programmer error; bail with
            // a clear failure rather than a silent broken state.
            return AssetState::Failed {
                reason: format!("no CharacterClass metadata for id {}", class_id),
            };
        }
    };

    // Flat graph: one clip node per label, all parented to the root
    // blend. Phase D will replace this with a blend-tree topology
    // (1D walk/run blend on speed, additive turn-in-place layer);
    // Phase B's job is just to make every clip *reachable* via a
    // stable `AnimationNodeIndex`.
    //
    // Per-node static weight is `1.0` — leaves the graph fully open
    // and lets `AnimationTransitions` (Phase D) drive blends entirely
    // through `ActiveAnimation::weight`. Leaving it at `0.0` would
    // multiply out the active weight and snap the rig back to bind
    // pose between clip switches. The root blend node normalises
    // contributions, so an inactive clip (no `AnimationPlayer.start`
    // called) contributes nothing regardless of static weight.
    let mut graph = AnimationGraph::new();
    let root = graph.root;
    let mut clip_nodes: [Option<AnimationNodeIndex>; ClipLabel::COUNT] =
        [None; ClipLabel::COUNT];
    for (label, handle) in clips {
        let node = graph.add_clip(handle.clone(), 1.0, root);
        clip_nodes[label.index()] = Some(node);
    }

    let graph_handle = graphs.add(graph);

    let bone_count = expected_bone_count(class);
    info!(
        class = class.name,
        id = class.id,
        bones = bone_count,
        animations = clips.len(),
        "character asset: Loaded {} ({} bones, {} animations)",
        class.name,
        bone_count,
        clips.len()
    );

    AssetState::Ready(ClassAssets {
        class,
        scene,
        graph: graph_handle,
        clip_nodes,
    })
}

/// Best-effort bone-count estimate before scene instantiation. We
/// don't have access to the parsed glTF skin here (Bevy hides it
/// inside the Scene asset), so we report the expected count from the
/// class definition: `head + neck + 2 feet + fp_cull_bones` — close
/// enough for the diagnostic line. Phase C does the authoritative
/// validation against the actual scene hierarchy on first spawn.
fn expected_bone_count(class: &voxeldust_core::character::CharacterClass) -> usize {
    use std::collections::HashSet;
    let mut names: HashSet<&str> = HashSet::new();
    names.insert(class.head_bone);
    names.insert(class.neck_bone);
    names.insert(class.left_foot_bone);
    names.insert(class.right_foot_bone);
    for b in class.fp_cull_bones {
        names.insert(b);
    }
    names.len()
}

