//! Client → server signal publish bridge.
//!
//! Publisher HUD widgets (buttons, toggles, sliders painted on in-world
//! HUD tiles) fire `PublishSignalEvent`. This plugin drains those
//! events and forwards them as `ClientMsg::SignalPublish` over the
//! primary shard's TCP. Server-side, the receiving shard validates
//! `publish_policy` against the sender's player_id before accepting.

use bevy::prelude::*;

use voxeldust_core::client_message::{ClientMsg, SignalPublishData};
use voxeldust_core::signal::types::SignalValue;

use crate::net::TcpSender;

/// Fire to publish a signal value to a channel. Server validates
/// `publish_policy` — publishes that fail are silently logged server
/// side (no client-facing error; future iteration adds a reject
/// broadcast).
#[derive(Message, Debug, Clone)]
pub struct PublishSignalEvent {
    pub channel: String,
    pub value: SignalValue,
}

pub struct SignalPublishPlugin;

impl Plugin for SignalPublishPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PublishSignalEvent>()
            .add_systems(Update, drain_publishes);
    }
}

fn drain_publishes(mut events: MessageReader<PublishSignalEvent>, tcp: Res<TcpSender>) {
    for ev in events.read() {
        let msg = ClientMsg::SignalPublish(SignalPublishData {
            channel_name: ev.channel.clone(),
            value: ev.value,
        });
        let serialized = msg.serialize();
        if tcp.tx.send(serialized).is_err() {
            tracing::warn!(channel = %ev.channel, "SignalPublish: TCP closed");
        }
    }
}
