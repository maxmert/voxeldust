//! Coverage + robustness for the peer-addr auto-resolver bin wiring
//! (`vd_bins::spawn_peer_resolver_if_configured`): the absent no-op, a present-valid spawn, and the two
//! fail-loud arms (a malformed `VD_PEER_HOSTS`, and a host for a peer that is not booked in `VD_PEERS`).
//! The helper's execution inside the SUBPROCESS bins (dev_cluster_smoke) is not llvm-captured, so these
//! in-binary tests exercise every arm directly.

use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use vd_core::NodeId;
use vd_io_prod::mesh::{MeshConfig, MeshControl, spawn_mesh};
use vd_io_prod::runtime::EnvConfig;
use vd_io_prod::trust::ClusterTrust;

/// An `EnvConfig` from literal pairs (the public map constructor).
fn env(pairs: &[(&str, &str)]) -> EnvConfig {
    EnvConfig::new(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
    )
}

/// A throwaway mesh node whose `Arc<MeshControl>` the helper needs. The runtime is returned so it (and the
/// endpoint) outlive the helper call.
fn node() -> (tokio::runtime::Runtime, Arc<MeshControl>) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("rt");
    let trust = ClusterTrust::generate("vd-peer-resolver-wiring").expect("trust");
    let addr = UdpSocket::bind("127.0.0.1:0")
        .expect("reserve")
        .local_addr()
        .expect("addr");
    let book: BTreeMap<NodeId, SocketAddr> = [(NodeId(2), addr)].into();
    let (_tx, ctl) = spawn_mesh(
        rt.handle(),
        &trust,
        &MeshConfig::new(NodeId(1), addr, book, 64, 0, vd_bins::world_generation()),
        None,
    )
    .expect("mesh");
    (rt, Arc::new(ctl))
}

#[test]
fn absent_vd_peer_hosts_is_a_noop() {
    let (rt, ctl) = node();
    // No VD_PEER_HOSTS ⇒ Ok, no task spawned (byte-identical to today — the outbox None-path discipline).
    vd_bins::spawn_peer_resolver_if_configured(
        &env(&[]),
        rt.handle(),
        ctl,
        Arc::new(AtomicBool::new(true)),
    )
    .expect("absent VD_PEER_HOSTS is a clean no-op");
}

#[test]
fn present_valid_vd_peer_hosts_spawns_the_resolver() {
    let (rt, ctl) = node();
    // Peer 2 is booked (VD_PEERS) + listed in VD_PEER_HOSTS ⇒ the resolver spawns. shutdown=true so the task
    // exits before any getaddrinfo (the while-condition checks shutdown first).
    let e = env(&[
        ("VD_PEERS", "2=127.0.0.1:9001"),
        ("VD_PEER_HOSTS", "2=vd-x.svc:9000"),
    ]);
    vd_bins::spawn_peer_resolver_if_configured(
        &e,
        rt.handle(),
        ctl,
        Arc::new(AtomicBool::new(true)),
    )
    .expect("a booked, well-formed VD_PEER_HOSTS spawns the resolver");
}

#[test]
fn malformed_vd_peer_hosts_fails_loud() {
    let (rt, ctl) = node();
    // A missing '=' is malformed ⇒ Err (never a silent mis-parse).
    let e = env(&[("VD_PEER_HOSTS", "2:vd-x.svc:9000")]);
    assert!(
        vd_bins::spawn_peer_resolver_if_configured(
            &e,
            rt.handle(),
            ctl,
            Arc::new(AtomicBool::new(true))
        )
        .is_err(),
        "a malformed VD_PEER_HOSTS must fail loud"
    );
}

#[test]
fn unbooked_peer_in_vd_peer_hosts_fails_loud() {
    let (rt, ctl) = node();
    // Peer 9 is NOT in VD_PEERS ⇒ the drift guard fails loud (the auto-resolver only re-plumbs booked peers).
    let e = env(&[
        ("VD_PEERS", "2=127.0.0.1:9001"),
        ("VD_PEER_HOSTS", "9=vd-y.svc:9000"),
    ]);
    assert!(
        vd_bins::spawn_peer_resolver_if_configured(
            &e,
            rt.handle(),
            ctl,
            Arc::new(AtomicBool::new(true))
        )
        .is_err(),
        "a VD_PEER_HOSTS peer not in VD_PEERS must fail loud"
    );
}
