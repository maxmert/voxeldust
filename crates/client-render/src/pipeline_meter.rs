//! ★ THE PIPELINE METER (G-NO-STALL, the temporal path's T2; ruling B33, 2026-10-02): how many of the engine's
//! cached pipelines are READY and how many still WAIT (queued or compiling), counted in the render world at the end of
//! each frame and shared with the main world, which writes both counts on the stamp's frame-cost row.
//!
//! G-NO-STALL asks for no pipeline compiled after the pre-roll: a ready count that grows after the pre-roll, or a
//! waiting count above nought, is a pipeline the frame had to wait for or draw without. The meter is a READING for the
//! dev-control harness — it changes nothing the game does.
//!
//! **Example.** The pilot logs in at the ground stand: the ready count climbs while the first frames compile the ground's,
//! the sky's and the reconstruction's pipelines, then holds. The pilot makes the window full screen: if the count holds
//! and nothing waits, that switch compiled nothing; if it rises by two, two pipelines compiled in the middle of the play.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::prelude::*;
use bevy::render::render_resource::{CachedPipelineState, PipelineCache};
use bevy::render::{Render, RenderApp, RenderSystems};

/// The two counts, shared by the two worlds: the ready pipelines, and the waiting ones.
#[derive(Resource, Clone, Default)]
pub(crate) struct PipelineMeter(Arc<[AtomicU64; 2]>);

impl PipelineMeter {
    /// The newest counts: `(ready, waiting)`.
    pub(crate) fn read(&self) -> (u64, u64) {
        (
            self.0[0].load(Ordering::Relaxed),
            self.0[1].load(Ordering::Relaxed),
        )
    }
}

/// Count the cache's pipelines by state, after the frame's work (so a pipeline this frame queued is counted).
fn count_pipelines(cache: Res<PipelineCache>, meter: Res<PipelineMeter>) {
    let (mut ready, mut waiting) = (0u64, 0u64);
    for pipeline in cache.pipelines() {
        match pipeline.state {
            CachedPipelineState::Ok(_) => ready += 1,
            CachedPipelineState::Queued | CachedPipelineState::Creating(_) => waiting += 1,
            CachedPipelineState::Err(_) => {}
        }
    }
    meter.0[0].store(ready, Ordering::Relaxed);
    meter.0[1].store(waiting, Ordering::Relaxed);
}

/// The meter's plugin: the shared counts in both worlds and the render world's count.
pub(crate) struct PipelineMeterPlugin;

impl Plugin for PipelineMeterPlugin {
    fn build(&self, app: &mut App) {
        let meter = PipelineMeter::default();
        app.insert_resource(meter.clone());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(meter)
            .add_systems(Render, count_pipelines.in_set(RenderSystems::Cleanup));
    }
}
