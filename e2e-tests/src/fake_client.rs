use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use voxeldust_core::client_message::{ClientMsg, ServerMsg};
use voxeldust_core::shard_message::MessageError;

/// A simulated game client for E2E testing.
pub struct FakeClient {
    stream: TcpStream,
}

impl FakeClient {
    /// Connect to a TCP address (gateway or shard).
    pub async fn connect(addr: SocketAddr) -> Result<Self, std::io::Error> {
        let stream = TcpStream::connect(addr).await?;
        Ok(Self { stream })
    }

    /// Send a Connect message.
    pub async fn send_connect(&mut self, player_name: &str) -> Result<(), std::io::Error> {
        let msg = ClientMsg::Connect {
            player_name: player_name.to_string(),
        };
        let data = msg.serialize();
        let len = (data.len() as u32).to_be_bytes();
        self.stream.write_all(&len).await?;
        self.stream.write_all(&data).await?;
        self.stream.flush().await?;
        Ok(())
    }

    /// Read a ServerMsg response.
    pub async fn recv_server_msg(&mut self) -> Result<ServerMsg, Box<dyn std::error::Error>> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf).await?;

        ServerMsg::deserialize(&buf).map_err(|e| e.into())
    }
}
