//! The day-1 cluster trust model, formalized (`docs/design/identity_persistence.md`
//! §5): ONE static cluster secret — a bundle of {cluster CA, node certificate, node
//! key} — distributed to every node; all inter-process QUIC is MUTUAL TLS against it.
//!
//! - Both directions verify: the server requires and validates the CLIENT certificate,
//!   the client validates the server certificate — against the same cluster CA.
//! - There is NO skip-verification path anywhere in this module, by construction
//!   (audit finding R7's enabler is unrepresentable).
//! - Per-node SPIFFE-style certificates from a real CA are deferred to an actual
//!   multi-tenant boundary; the SEAM (this type) stays the same when they land (P3+).

use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::WebPkiClientVerifier;

/// ALPN tag pinning the inter-shard protocol on every cluster connection.
pub const INTERSHARD_ALPN: &[u8] = b"vd-intershard/1";

/// Trust setup failures (operator/config errors — loud and typed).
#[derive(Debug, thiserror::Error)]
pub enum TrustError {
    #[error("certificate generation: {0}")]
    Generate(String),
    #[error("tls config: {0}")]
    Tls(String),
}

/// The single cluster secret. Serialize the three DER blobs into the cluster's
/// secret store; every node loads the same bundle via [`ClusterTrust::from_der`].
#[derive(Clone)]
pub struct ClusterTrust {
    ca_cert_der: Vec<u8>,
    node_cert_der: Vec<u8>,
    node_key_pkcs8: Vec<u8>,
}

impl std::fmt::Debug for ClusterTrust {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The node key is SECRET: never let a debug log leak it.
        f.debug_struct("ClusterTrust")
            .field("ca_cert_der_len", &self.ca_cert_der.len())
            .field("node_cert_der_len", &self.node_cert_der.len())
            .field("node_key_pkcs8", &"<redacted>")
            .finish()
    }
}

impl ClusterTrust {
    /// Generate a fresh cluster bundle: a self-signed cluster CA and one node
    /// certificate (SANs: `cluster_name`, `localhost`) signed by it.
    ///
    /// # Errors
    /// [`TrustError::Generate`] on certificate-generation failure.
    pub fn generate(cluster_name: &str) -> Result<ClusterTrust, TrustError> {
        let gen_err = |e: rcgen::Error| TrustError::Generate(e.to_string());

        let mut ca_params = rcgen::CertificateParams::new(Vec::<String>::new()).map_err(gen_err)?;
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        ca_params.distinguished_name.push(
            rcgen::DnType::CommonName,
            format!("{cluster_name} cluster CA"),
        );
        let ca_key = rcgen::KeyPair::generate().map_err(gen_err)?;
        let ca_cert = ca_params.self_signed(&ca_key).map_err(gen_err)?;

        let mut node_params =
            rcgen::CertificateParams::new(vec![cluster_name.to_owned(), "localhost".to_owned()])
                .map_err(gen_err)?;
        node_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, format!("{cluster_name} node"));
        node_params.extended_key_usages = vec![
            rcgen::ExtendedKeyUsagePurpose::ServerAuth,
            rcgen::ExtendedKeyUsagePurpose::ClientAuth,
        ];
        let node_key = rcgen::KeyPair::generate().map_err(gen_err)?;
        let node_cert = node_params
            .signed_by(&node_key, &ca_cert, &ca_key)
            .map_err(gen_err)?;

        Ok(ClusterTrust {
            ca_cert_der: ca_cert.der().to_vec(),
            node_cert_der: node_cert.der().to_vec(),
            node_key_pkcs8: node_key.serialize_der(),
        })
    }

    /// Load the bundle from its three DER blobs (the deployed cluster secret).
    #[must_use]
    pub fn from_der(
        ca_cert_der: Vec<u8>,
        node_cert_der: Vec<u8>,
        node_key_pkcs8: Vec<u8>,
    ) -> ClusterTrust {
        ClusterTrust {
            ca_cert_der,
            node_cert_der,
            node_key_pkcs8,
        }
    }

    /// The three DER blobs, for writing into the cluster's secret store.
    /// Order: (ca_cert, node_cert, node_key_pkcs8) — the key is SECRET.
    #[must_use]
    pub fn to_der_parts(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            self.ca_cert_der.clone(),
            self.node_cert_der.clone(),
            self.node_key_pkcs8.clone(),
        )
    }

    /// Write the bundle into a directory as three DER files (`ca.der`,
    /// `node.der`, `key.der`) — the dev/test cluster-secret distribution.
    ///
    /// # Errors
    /// Filesystem failures.
    pub fn write_der_dir(&self, dir: &std::path::Path) -> Result<(), std::io::Error> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("ca.der"), &self.ca_cert_der)?;
        std::fs::write(dir.join("node.der"), &self.node_cert_der)?;
        std::fs::write(dir.join("key.der"), &self.node_key_pkcs8)?;
        Ok(())
    }

    /// Load a bundle written by [`ClusterTrust::write_der_dir`].
    ///
    /// # Errors
    /// Filesystem failures (missing/unreadable files).
    pub fn from_der_dir(dir: &std::path::Path) -> Result<ClusterTrust, std::io::Error> {
        Ok(ClusterTrust {
            ca_cert_der: std::fs::read(dir.join("ca.der"))?,
            node_cert_der: std::fs::read(dir.join("node.der"))?,
            node_key_pkcs8: std::fs::read(dir.join("key.der"))?,
        })
    }

    fn roots(&self) -> Result<rustls::RootCertStore, TrustError> {
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(CertificateDer::from(self.ca_cert_der.clone()))
            .map_err(|e| TrustError::Tls(e.to_string()))?;
        Ok(roots)
    }

    fn chain(&self) -> Vec<CertificateDer<'static>> {
        vec![
            CertificateDer::from(self.node_cert_der.clone()),
            CertificateDer::from(self.ca_cert_der.clone()),
        ]
    }

    fn key(&self) -> PrivateKeyDer<'static> {
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(self.node_key_pkcs8.clone()))
    }

    /// quinn server config: presents the node certificate AND requires a client
    /// certificate chained to the cluster CA (mutual TLS).
    ///
    /// # Errors
    /// [`TrustError::Tls`] on TLS-config assembly failure.
    pub fn quinn_server_config(&self) -> Result<quinn::ServerConfig, TrustError> {
        let tls_err = |e: rustls::Error| TrustError::Tls(e.to_string());
        let verifier = WebPkiClientVerifier::builder(Arc::new(self.roots()?))
            .build()
            .map_err(|e| TrustError::Tls(e.to_string()))?;
        let mut tls = rustls::ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(self.chain(), self.key())
            .map_err(tls_err)?;
        tls.alpn_protocols = vec![INTERSHARD_ALPN.to_vec()];
        let quic = quinn::crypto::rustls::QuicServerConfig::try_from(tls)
            .map_err(|e| TrustError::Tls(e.to_string()))?;
        Ok(quinn::ServerConfig::with_crypto(Arc::new(quic)))
    }

    /// quinn client config: verifies the server against the cluster CA AND presents
    /// the node certificate as the client identity (mutual TLS).
    ///
    /// # Errors
    /// [`TrustError::Tls`] on TLS-config assembly failure.
    pub fn quinn_client_config(&self) -> Result<quinn::ClientConfig, TrustError> {
        let tls_err = |e: rustls::Error| TrustError::Tls(e.to_string());
        let mut tls = rustls::ClientConfig::builder()
            .with_root_certificates(self.roots()?)
            .with_client_auth_cert(self.chain(), self.key())
            .map_err(tls_err)?;
        tls.alpn_protocols = vec![INTERSHARD_ALPN.to_vec()];
        let quic = quinn::crypto::rustls::QuicClientConfig::try_from(tls)
            .map_err(|e| TrustError::Tls(e.to_string()))?;
        Ok(quinn::ClientConfig::new(Arc::new(quic)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime")
    }

    /// Attempt a full QUIC handshake: a server under `server_trust` and a client
    /// under `client_trust`. Returns whether BOTH sides completed.
    fn handshake(server_trust: &ClusterTrust, client_trust: &ClusterTrust) -> bool {
        let rt = runtime();
        let server_config = server_trust.quinn_server_config().expect("server config");
        let client_config = client_trust.quinn_client_config().expect("client config");
        rt.block_on(async move {
            let bind: SocketAddr = "127.0.0.1:0".parse().expect("addr");
            let server = quinn::Endpoint::server(server_config, bind).expect("server endpoint");
            let server_addr = server.local_addr().expect("addr");
            let accept = tokio::spawn(async move {
                let incoming = server.accept().await?;
                incoming.await.ok()
            });

            let mut client = quinn::Endpoint::client(bind).expect("client endpoint");
            client.set_default_client_config(client_config);
            let client_side = match client.connect(server_addr, "localhost") {
                Ok(connecting) => connecting.await.ok(),
                Err(_) => None,
            };
            let server_side = accept.await.ok().flatten();
            match (client_side, server_side) {
                (Some(_), Some(server_conn)) => {
                    // MUTUAL: the server saw and validated a client certificate.
                    let identity = server_conn.peer_identity();
                    identity.is_some()
                }
                _ => false,
            }
        })
    }

    #[test]
    fn mtls_handshake_succeeds_with_the_shared_bundle() {
        let trust = ClusterTrust::generate("vd-test-cluster").expect("generate");
        assert!(
            handshake(&trust, &trust),
            "shared cluster trust must handshake mutually"
        );
    }

    #[test]
    fn a_foreign_cluster_is_rejected_in_both_directions() {
        let ours = ClusterTrust::generate("vd-test-cluster").expect("generate");
        let theirs = ClusterTrust::generate("vd-evil-cluster").expect("generate");
        assert!(
            !handshake(&ours, &theirs),
            "a client from another cluster must be rejected"
        );
        assert!(
            !handshake(&theirs, &ours),
            "a server from another cluster must be rejected"
        );
    }

    #[test]
    fn der_roundtrip_preserves_the_working_bundle() {
        let trust = ClusterTrust::generate("vd-test-cluster").expect("generate");
        let (ca, cert, key) = trust.to_der_parts();
        let reloaded = ClusterTrust::from_der(ca, cert, key);
        assert!(
            handshake(&trust, &reloaded),
            "a reloaded bundle is the same trust"
        );
    }

    #[test]
    fn der_dir_roundtrip_preserves_the_bundle() {
        let trust = ClusterTrust::generate("vd-test-cluster").expect("generate");
        let dir = std::env::temp_dir().join(format!("vd-trust-{}", std::process::id()));
        trust.write_der_dir(&dir).expect("write");
        let reloaded = ClusterTrust::from_der_dir(&dir).expect("read");
        assert!(
            handshake(&trust, &reloaded),
            "same trust after the round trip"
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
        assert!(
            ClusterTrust::from_der_dir(&dir).is_err(),
            "missing dir is loud"
        );
    }

    #[test]
    fn debug_never_prints_the_key() {
        let trust = ClusterTrust::generate("vd-test-cluster").expect("generate");
        let rendered = format!("{trust:?}");
        assert!(rendered.contains("<redacted>"));
        // No raw key byte rendering sneaks in.
        assert!(!rendered.contains("node_key_pkcs8: ["));
    }
}
