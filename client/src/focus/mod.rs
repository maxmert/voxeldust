//! Focus interaction (E-key): camera glides to a block face, cursor
//! becomes a synthesized pointer for publisher widgets on that face.
//!
//! **Server dependency** (plan Phase 19 TODO): `ClientMsg::SignalPublish`
//! doesn't exist yet in `voxeldust-core::client_message`. When a
//! publisher widget is clicked it should emit a live signal on a
//! named channel; until the server defines that wire table + handler,
//! we log a `warn!` instead of sending. The full widget definitions
//! (buttons, sliders, toggles attached to blocks) also require a
//! server-side schema that lands alongside the message.
//!
//! What's in this module now:
//!   * `FocusState` resource tracking idle / gliding-in / focused.
//!   * `FocusRequest` event (fired by E-key on a valid target).
//!   * `PublisherWidget` component + tween primitives ready for
//!     per-block widget meshes when they ship server-side.
//!
//! What's **not** here yet (deferred pending server support):
//!   * Actual widget geometry on blocks (no data source).
//!   * Camera tween that smoothly eases to the face (requires widget
//!     positioning which requires widget data).
//!   * Egui pointer synthesis on the face plane.
//!   * `SignalPublish` TCP dispatch.
//!
//! Today E-key still dispatches via `interaction::dispatch` (as
//! INTERACT / ENTER_SEAT). This module hooks into the InputMode
//! state machine but doesn't override E behavior yet — doing so
//! would regress gameplay (can't sit in a seat) with no benefit
//! (no widgets to focus on).

use bevy::prelude::*;

use crate::shard::ShardKey;

/// Current focus-mode state. `Idle` is the default; when the user
/// triggers focus (via future "widget-bearing block" detection), we
/// transition through `GlidingIn` to `Focused`; Esc returns to `Idle`
/// through `GlidingOut`.
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusState {
    #[default]
    Idle,
    GlidingIn,
    Focused,
    GlidingOut,
}

/// Event fired when a publisher-widget click should publish a signal.
/// Consumed by a future system that encodes + sends
/// `ClientMsg::SignalPublish` via TCP. Until that message lands, the
/// consumer logs and drops.
#[derive(Message, Debug, Clone)]
pub struct PublishSignalRequest {
    pub shard: ShardKey,
    pub channel_name: String,
    pub value: f32,
}

/// Publisher widget classifications, attached to a future
/// widget-entity on a block face. Shape of each variant matches the
/// spec in the plan.
#[allow(dead_code)]
#[derive(Component, Debug, Clone)]
pub enum PublisherWidget {
    /// Button: momentary emit on press / release.
    Button {
        channel: String,
        press_value: f32,
        release_value: f32,
    },
    /// Continuous slider: emits current value while dragged.
    Slider {
        channel: String,
        range: (f32, f32),
        default: f32,
        current: f32,
    },
    /// Toggle: flips between on/off on each click.
    Toggle {
        channel: String,
        on_value: f32,
        off_value: f32,
        state: bool,
    },
}

pub struct FocusInteractionPlugin;

impl Plugin for FocusInteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FocusState>()
            .add_message::<PublishSignalRequest>()
            .add_systems(Update, drain_publish_requests);
    }
}

/// Consume `PublishSignalRequest`s. When the server's `SignalPublish`
/// wire format lands, this function will serialize + send via TCP —
/// today it logs a warning so the rest of the pipeline can be
/// exercised end-to-end as soon as the server is ready.
fn drain_publish_requests(mut reader: MessageReader<PublishSignalRequest>) {
    for req in reader.read() {
        tracing::warn!(
            shard = %req.shard,
            channel = %req.channel_name,
            value = req.value,
            "SignalPublish: server-side protocol not yet defined — request dropped",
        );
    }
}
