use std::net::SocketAddr;

use bevy::ecs::resource::Resource;
use clap::Parser;

#[derive(Parser, Resource, Debug, Clone)]
#[command(name = "client", about = "Bevy-powered voxeldust client")]
pub struct Cli {
    /// Gateway address to connect to.
    #[arg(long, default_value = "127.0.0.1:7777")]
    pub gateway: SocketAddr,

    /// Display name announced to the server.
    #[arg(long, default_value = "Player")]
    pub name: String,
}
