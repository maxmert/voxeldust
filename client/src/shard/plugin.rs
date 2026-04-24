//! The `ShardTypePlugin` trait — one implementation per shard-type (SHIP,
//! SYSTEM, PLANET, GALAXY, future DEBRIS / STATION / …). Client-core never
//! matches on specific shard-types; every per-type decision routes
//! through this trait.
//!
//! Method surface grows phase-by-phase: Phase 3 defines identity +
//! lifecycle. Phase 4 adds coordinate mapping. Phase 5 adds vertex
//! transforms. Phase 10 adds transition hooks. Phase 11 adds camera
//! frames. Phase 14 adds raycast contributions. Phase 15 adds
//! interaction schemas.
//!
//! All methods after identity + lifecycle ship with **sensible defaults**
//! so a minimal plugin (just name + shard_type) compiles. Later phases
//! override defaults to wire up full behavior.

use std::time::Duration;

use bevy::prelude::*;
use glam::DVec3;

use voxeldust_core::client_message::WorldStateData;

use crate::shard::runtime::ShardRuntime;

// `ShardRuntime` is still used in the body of the trait (coordinate
// mapping). The HUD context below uses DVec3 directly to avoid
// requiring access to `Secondaries` (which doesn't carry the primary).

/// Trait implemented once per shard-type. Concrete impls live in
/// `crate::shard_types::<name>.rs` and are registered into the
/// `ShardTypeRegistry` at app-build time.
pub trait ShardTypePlugin: Send + Sync + 'static {
    // ────────────────────────────────────────────────────────────────
    // Identity
    // ────────────────────────────────────────────────────────────────

    /// Wire-level shard_type key (u8 from the protocol). Must be unique
    /// across all registered plugins.
    fn shard_type(&self) -> u8;

    /// Display name for HUD + tracing.
    fn name(&self) -> &'static str;

    /// "Scene-context" shard-types persist across primary flips (SYSTEM,
    /// GALAXY today). Network layer is authoritative; client uses this
    /// as a hint for HUD / telemetry.
    fn is_scene_context(&self) -> bool {
        false
    }

    // ────────────────────────────────────────────────────────────────
    // Lifecycle
    // ────────────────────────────────────────────────────────────────

    /// Called once per shard-instance spawn, on the root `ChunkSource`
    /// entity. The plugin may insert per-shard components, register
    /// per-shard resources, or do any other scoped setup.
    fn on_shard_spawn(&self, _cmds: &mut Commands, _shard: &ShardRuntime) {}

    /// Called once per shard-instance despawn, before the entity is
    /// recursively despawned.
    fn on_shard_despawn(&self, _cmds: &mut Commands, _shard: &ShardRuntime) {}

    // ────────────────────────────────────────────────────────────────
    // Coordinate mapping (Phase 4)
    // ────────────────────────────────────────────────────────────────

    /// Shard-local → system-space. Default: identity (correct for SYSTEM;
    /// SHIP / PLANET / GALAXY override).
    fn to_system_space(&self, shard: &ShardRuntime, local: DVec3) -> DVec3 {
        shard.origin + shard.rotation * local
    }

    /// System-space → shard-local (inverse of `to_system_space`). Default
    /// matches the inverse of the default above.
    fn from_system_space(&self, shard: &ShardRuntime, world: DVec3) -> DVec3 {
        shard.rotation.inverse() * (world - shard.origin)
    }

    // ────────────────────────────────────────────────────────────────
    // Transitions (Phase 10)
    // ────────────────────────────────────────────────────────────────

    /// Does this shard-type want its last primary's ChunkSource retained
    /// in the grace window after a departure? Default: yes.
    fn wants_grace(&self) -> bool {
        true
    }

    /// How long to retain graced sources. Default: 1.5 s.
    fn grace_duration(&self) -> Duration {
        Duration::from_millis(1500)
    }

    // ────────────────────────────────────────────────────────────────
    // HUD (Phase 21 / H9)
    // ────────────────────────────────────────────────────────────────

    /// Per-shard-type HUD contribution, rendered when this shard is the
    /// **primary**. Returns `(label, value)` pairs for the connection /
    /// status panel. Default: empty — client-core renders nothing for
    /// this plugin's section. Impls supply pilot data (SHIP), orbital
    /// elements (SYSTEM), altitude / biome (PLANET), warp ETA (GALAXY).
    fn hud_summary(&self, _ctx: &HudSummaryCtx) -> Vec<(String, String)> {
        Vec::new()
    }

    // ────────────────────────────────────────────────────────────────
    // Future hooks (declared as defaults; wired in later phases)
    // ────────────────────────────────────────────────────────────────

    // fn primary_camera_frame(&self, ctx: &CameraCtx) -> CameraFrame;
    //     ↑ Phase 11
    // fn mesh_vertex_transform(&self) -> Option<VertexTransformFn>;
    //     ↑ Phase 5
    // fn raycast_contribution(&self, shard: &ShardRuntime) -> Vec<RaycastTarget>;
    //     ↑ Phase 14
    // fn interaction_schema(
    //     &self, shard: &ShardRuntime, kind: FunctionalBlockKind,
    // ) -> Option<&'static InteractionSchema>;
    //     ↑ Phase 15
    // fn depart(&self, ctx: &DepartureCtx) -> DepartureContext;
    // fn arrive(&self, src: &DepartureContext, ctx: &ArrivalCtx) -> ArrivalPose;
    //     ↑ Phase 10
    // fn render_contribution(&self, app: &mut App);
    //     ↑ Phase 6+ (per-plugin contribution; wired via `bevy::Plugin::build`)
}

/// Context the `hud_summary` trait method reads from. Borrowed across
/// every plugin call per frame so the HUD panel is cheap to rebuild.
/// Holds the primary-shard's f64 origin directly rather than the full
/// `ShardRuntime` so the HUD pass doesn't need a borrow of
/// `Secondaries` (primary isn't stored there).
pub struct HudSummaryCtx<'a> {
    pub shard_origin: DVec3,
    pub primary_ws: Option<&'a WorldStateData>,
    pub camera_world: DVec3,
}
