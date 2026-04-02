use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use voxeldust_core::shard_message::{MessageError, ShardMsg};
use voxeldust_core::shard_types::ShardId;

use crate::circuit_breaker::CircuitBreaker;

/// Timeout for QUIC connect and send operations.
const OPERATION_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum message size (64 KB).
const MAX_MESSAGE_SIZE: usize = 65_536;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("QUIC connection error: {0}")]
    Connection(#[from] quinn::ConnectionError),
    #[error("QUIC connect error: {0}")]
    Connect(#[from] quinn::ConnectError),
    #[error("write error: {0}")]
    Write(#[from] quinn::WriteError),
    #[error("read error: {0}")]
    Read(#[from] quinn::ReadExactError),
    #[error("message error: {0}")]
    Message(#[from] MessageError),
    #[error("circuit breaker open for peer {0}")]
    CircuitBreakerOpen(ShardId),
    #[error("operation timed out")]
    Timeout,
    #[error("message too large: {size} bytes (max {MAX_MESSAGE_SIZE})")]
    MessageTooLarge { size: usize },
    #[error("TLS error: {0}")]
    Tls(String),
    #[error("read to end error: {0}")]
    ReadToEnd(#[from] quinn::ReadToEndError),
}

/// A persistent connection + stream to a peer shard.
struct PeerChannel {
    conn: quinn::Connection,
    /// Persistent unidirectional stream — reused across messages to avoid
    /// ~100-200μs overhead of opening a new stream per message.
    send_stream: Option<SendStream>,
}

/// Manages QUIC connections to peer shards.
pub struct QuicTransport {
    endpoint: Endpoint,
    /// Active connections to peer shards, keyed by ShardId.
    peers: Mutex<HashMap<ShardId, PeerChannel>>,
    /// Per-peer circuit breakers.
    breakers: Mutex<HashMap<ShardId, CircuitBreaker>>,
}

impl QuicTransport {
    /// Create a new transport that listens on the given address.
    pub async fn bind(listen_addr: SocketAddr) -> Result<Arc<Self>, TransportError> {
        let (server_config, client_config) = generate_self_signed_config()
            .map_err(|e| TransportError::Tls(e.to_string()))?;

        let endpoint = Endpoint::server(server_config, listen_addr)
            .map_err(|e| TransportError::Tls(e.to_string()))?;

        // Set default client config for outbound connections.
        let mut ep = endpoint;
        ep.set_default_client_config(client_config);

        Ok(Arc::new(Self {
            endpoint: ep,
            peers: Mutex::new(HashMap::new()),
            breakers: Mutex::new(HashMap::new()),
        }))
    }

    /// Create a transport from an existing endpoint (useful for testing).
    pub fn from_endpoint(endpoint: Endpoint) -> Arc<Self> {
        Arc::new(Self {
            endpoint,
            peers: Mutex::new(HashMap::new()),
            breakers: Mutex::new(HashMap::new()),
        })
    }

    /// Get the local address this transport is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.endpoint.local_addr().unwrap()
    }

    /// Send a message to a peer shard. Reuses a persistent QUIC stream per peer.
    pub async fn send(
        &self,
        peer_id: ShardId,
        peer_addr: SocketAddr,
        msg: &ShardMsg,
    ) -> Result<(), TransportError> {
        // Check circuit breaker.
        {
            let breakers = self.breakers.lock().await;
            if let Some(cb) = breakers.get(&peer_id) {
                if !cb.allow_request() {
                    return Err(TransportError::CircuitBreakerOpen(peer_id));
                }
            }
        }

        let result = tokio::time::timeout(OPERATION_TIMEOUT, async {
            let data = msg.serialize();
            if data.len() > MAX_MESSAGE_SIZE {
                return Err(TransportError::MessageTooLarge { size: data.len() });
            }

            let mut peers = self.peers.lock().await;
            let channel = self.get_or_connect_channel(&mut peers, peer_id, peer_addr).await?;

            // Get or open a persistent send stream.
            if channel.send_stream.is_none() {
                channel.send_stream = Some(channel.conn.open_uni().await?);
            }
            let send = channel.send_stream.as_mut().unwrap();

            // Length-prefix: 4 bytes big-endian.
            let len_bytes = (data.len() as u32).to_be_bytes();
            let write_result = async {
                send.write_all(&len_bytes).await?;
                send.write_all(&data).await?;
                Ok::<(), quinn::WriteError>(())
            }.await;

            if let Err(e) = write_result {
                // Stream broke — discard it so next send reopens.
                channel.send_stream = None;
                return Err(TransportError::Write(e));
            }

            Ok::<(), TransportError>(())
        })
        .await;

        match result {
            Ok(Ok(())) => {
                self.record_success(peer_id).await;
                Ok(())
            }
            Ok(Err(e)) => {
                self.record_failure(peer_id).await;
                // Remove stale connection on error.
                self.peers.lock().await.remove(&peer_id);
                Err(e)
            }
            Err(_) => {
                self.record_failure(peer_id).await;
                self.peers.lock().await.remove(&peer_id);
                Err(TransportError::Timeout)
            }
        }
    }

    /// Accept the next incoming QUIC connection.
    pub async fn accept(&self) -> Option<IncomingConnection> {
        let incoming = self.endpoint.accept().await?;
        match incoming.await {
            Ok(conn) => {
                info!(remote = %conn.remote_address(), "accepted QUIC connection");
                Some(IncomingConnection { connection: conn })
            }
            Err(e) => {
                warn!(%e, "QUIC incoming connection handshake failed");
                None
            }
        }
    }

    /// Shut down the endpoint gracefully.
    pub fn shutdown(&self) {
        self.endpoint
            .close(quinn::VarInt::from_u32(0), b"shutdown");
    }

    async fn get_or_connect_channel<'a>(
        &self,
        peers: &'a mut HashMap<ShardId, PeerChannel>,
        peer_id: ShardId,
        peer_addr: SocketAddr,
    ) -> Result<&'a mut PeerChannel, TransportError> {
        // Check if existing connection is still alive.
        let needs_reconnect = match peers.get(&peer_id) {
            Some(ch) => ch.conn.close_reason().is_some(),
            None => true,
        };
        if needs_reconnect {
            peers.remove(&peer_id);
            // Open new connection.
            debug!(%peer_id, %peer_addr, "connecting to peer shard");
            let conn = self
                .endpoint
                .connect(peer_addr, "voxeldust-shard")?
                .await?;
            info!(%peer_id, %peer_addr, "connected to peer shard");
            peers.insert(peer_id, PeerChannel { conn, send_stream: None });
        }
        Ok(peers.get_mut(&peer_id).unwrap())
    }

    async fn record_success(&self, peer_id: ShardId) {
        let mut breakers = self.breakers.lock().await;
        if let Some(cb) = breakers.get_mut(&peer_id) {
            cb.record_success();
        }
    }

    async fn record_failure(&self, peer_id: ShardId) {
        let mut breakers = self.breakers.lock().await;
        let cb = breakers.entry(peer_id).or_default();
        cb.record_failure();
        if !cb.allow_request() {
            warn!(
                %peer_id,
                failures = cb.consecutive_failures(),
                "circuit breaker opened for peer"
            );
        }
    }
}

/// A QUIC connection accepted from a peer shard.
pub struct IncomingConnection {
    connection: quinn::Connection,
}

impl IncomingConnection {
    /// Accept a uni stream and read one length-prefixed message from it.
    /// Called in a loop; each call accepts the next stream from this connection.
    /// Supports both persistent streams (multiple messages per stream) and
    /// one-shot streams (one message per stream) via the `recv_loop` helper.
    pub async fn recv(&self) -> Result<ShardMsg, TransportError> {
        let mut recv: RecvStream = self.connection.accept_uni().await?;
        Self::read_one_message(&mut recv).await
    }

    /// Read a single length-prefixed message from a stream.
    async fn read_one_message(recv: &mut RecvStream) -> Result<ShardMsg, TransportError> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > MAX_MESSAGE_SIZE {
            return Err(TransportError::MessageTooLarge { size: len });
        }

        let mut data = vec![0u8; len];
        recv.read_exact(&mut data).await?;
        let msg = ShardMsg::deserialize(&data)?;
        Ok(msg)
    }

    /// Accept streams and read multiple messages per stream, forwarding to a channel.
    /// Handles persistent streams where a sender writes many length-prefixed messages
    /// on a single uni stream.
    pub async fn recv_loop(
        self,
        tx: tokio::sync::mpsc::UnboundedSender<ShardMsg>,
    ) -> Result<(), TransportError> {
        loop {
            let mut recv: RecvStream = self.connection.accept_uni().await?;
            let tx = tx.clone();
            // Spawn a task per stream to read all messages from it.
            tokio::spawn(async move {
                loop {
                    match Self::read_one_message(&mut recv).await {
                        Ok(msg) => {
                            if tx.send(msg).is_err() { return; }
                        }
                        Err(_) => return, // stream ended or error
                    }
                }
            });
        }
    }

    /// The remote address of this connection.
    pub fn remote_addr(&self) -> SocketAddr {
        self.connection.remote_address()
    }
}

/// Generate self-signed TLS config for inter-shard QUIC.
/// In production, use proper certificates. For development and inter-shard
/// communication within a cluster, self-signed is acceptable.
fn generate_self_signed_config() -> Result<(ServerConfig, ClientConfig), Box<dyn std::error::Error>>
{
    let key_pair = rcgen::KeyPair::generate()?;
    let cert_params = rcgen::CertificateParams::new(vec!["voxeldust-shard".to_string()])?;
    let cert = cert_params.self_signed(&key_pair)?;
    let cert_der = cert.der().clone();
    let key_der = key_pair.serialize_der();

    let cert_chain = vec![cert_der.clone()];

    // Server config
    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            cert_chain.clone(),
            rustls::pki_types::PrivatePkcs8KeyDer::from(key_der.clone()).into(),
        )?;
    server_crypto.alpn_protocols = vec![b"voxeldust".to_vec()];
    let server_config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)?,
    ));

    // Client config — skip server cert verification for inter-shard communication.
    // All shards are within the same trusted cluster.
    let mut client_crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![b"voxeldust".to_vec()];
    let client_config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)?,
    ));

    Ok((server_config, client_config))
}

/// Certificate verifier that accepts any server certificate.
/// Used for inter-shard QUIC where all endpoints are within a trusted cluster.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}

/// Create a pair of connected QUIC endpoints for testing.
/// Returns (endpoint_a, endpoint_b) bound to localhost on random ports.
pub async fn test_endpoint_pair() -> Result<(Arc<QuicTransport>, Arc<QuicTransport>), TransportError>
{
    let addr_a: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let addr_b: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let transport_a = QuicTransport::bind(addr_a).await?;
    let transport_b = QuicTransport::bind(addr_b).await?;

    Ok((transport_a, transport_b))
}
