use std::net::SocketAddr;

use bevy::ecs::resource::Resource;
use clap::Parser;

use crate::config::LightingPreset;

#[derive(Parser, Resource, Debug, Clone)]
#[command(name = "client", about = "Bevy-powered voxeldust client")]
pub struct Cli {
    /// Gateway address to connect to.
    #[arg(long, default_value = "127.0.0.1:7777")]
    pub gateway: SocketAddr,

    /// Display name announced to the server.
    #[arg(long, default_value = "Player")]
    pub name: String,

    /// Optional ship-grouping key. When set, the gateway hashes THIS
    /// instead of `--name` to derive the ship_id, so two clients with
    /// distinct names sharing the same `--ship-join` value land on
    /// the same ship shard. Default empty = legacy per-player ship.
    /// Used for multi-client playtests / "join my ship" flows.
    #[arg(long, default_value = "")]
    pub ship_join: String,

    /// Lighting / rendering fidelity preset. Affects shadows, AA,
    /// post-FX presence, atmosphere method, and IBL intensity — but
    /// **not** any lighting *value* (sun colour, illuminance,
    /// atmospheric scattering, eclipse occlusion are physics-derived
    /// and identical at every preset). Defaults to `Medium`. The full
    /// per-preset field breakdown lives in
    /// `client/src/config/graphics.rs::LightingFidelity::{low, medium,
    /// high, ultra}`.
    #[arg(long, value_enum, default_value_t = LightingPreset::Medium)]
    pub graphics_preset: LightingPreset,
}
