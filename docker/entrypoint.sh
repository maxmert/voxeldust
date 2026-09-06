#!/bin/sh
# S4b (cloud-ready k3d) — the k8s entrypoint. VD_PEERS is SocketAddr-ONLY (EnvConfig::peer_book parses
# id=IP:port, never a DNS name), so this DNS-resolves the three headless-Service pod names to literal IPv4
# addresses, builds the id=IP:port book for THIS role, then `exec`s the bin (so it becomes PID 1 and receives
# SIGTERM directly — the graceful-drain path; a non-exec shell would eat SIGTERM and force a SIGKILL past the
# grace, corrupting the orchestrator's un-fsynced redb). NodeIds: orch=1, gateway=2 (vd_bins::ORCH/
# GATEWAY). Usage: entrypoint.sh <vd-orchestrator|vd-gateway>
set -eu

NS="${VD_K8S_NAMESPACE:-voxeldust}"
DOM="svc.cluster.local"
MESH_PORT="${VD_MESH_PORT:-9000}"

# $1 = fqdn -> first IPv4 A record, retrying until the headless Service publishes the peer's endpoint.
# `ahostsv4` pins IPv4 (VD_BIND=0.0.0.0 is v4); publishNotReadyAddresses on the peer Services means a
# NotReady peer's A-record is published, so cold-start does not deadlock on mutual readiness.
resolve() {
  i=0
  while [ "$i" -lt 90 ]; do
    ip="$(getent ahostsv4 "$1" | awk '{ print $1; exit }')" || true
    if [ -n "${ip:-}" ]; then
      echo "$ip"
      return 0
    fi
    echo "entrypoint: waiting for DNS $1 (attempt $i)" >&2
    sleep 1
    i=$((i + 1))
  done
  echo "entrypoint: FATAL $1 never resolved after 90s" >&2
  return 1
}

ROLE="${1:-}"
# The peer FQDNs (headless-Service pod names). Resolved to IPs for VD_PEERS (dialed now) AND kept UN-resolved
# for VD_PEER_HOSTS — the in-process auto-resolver re-resolves these periodically and pushes a peer's NEW pod
# IP via update_peer_addr, so a rescheduled StatefulSet pod (same DNS name, new IP) is re-plumbed for INITIATED
# traffic (reply-on-connection only covers replies).
ORCH_FQDN="vd-orch-0.vd-orch.$NS.$DOM"
GW_FQDN="vd-gateway-0.vd-gateway.$NS.$DOM"
ORCH="$(resolve "$ORCH_FQDN")"
GW="$(resolve "$GW_FQDN")"

# VD_PEERS = id=IP:port (dialed now); VD_PEER_HOSTS = the SAME roster + NodeIds as id=fqdn:port (re-resolved).
case "$ROLE" in
  # THE DEMAND SHAPE (foundation slice 5, 2026-09-06): no static shard. The orchestrator forks realm
  # shards as its own child processes with an env it derives itself; only the two long-lived roles
  # book each other here.
  vd-orchestrator)
    VD_PEERS="2=$GW:$MESH_PORT"
    VD_PEER_HOSTS="2=$GW_FQDN:$MESH_PORT" ;;
  vd-gateway)
    VD_PEERS="1=$ORCH:$MESH_PORT"
    VD_PEER_HOSTS="1=$ORCH_FQDN:$MESH_PORT" ;;
  *) echo "entrypoint: FATAL unknown role '$ROLE' (expected vd-orchestrator|vd-gateway)" >&2; exit 1 ;;
esac

# GUARD: a present-but-empty/mis-shaped VD_PEERS silently parses to a SOLO node (peer_book drops empty entries),
# so fail loud rather than boot a partitioned singleton. The shape must contain at least one id=host:port. The
# SAME guard applies to VD_PEER_HOSTS (mis-shaped ⇒ the auto-resolver's fail-loud drift guard, but catch it here).
case "$VD_PEERS" in
  *=*:*) : ;;
  *) echo "entrypoint: FATAL empty/mis-shaped VD_PEERS='$VD_PEERS'" >&2; exit 1 ;;
esac
case "$VD_PEER_HOSTS" in
  *=*:*) : ;;
  *) echo "entrypoint: FATAL empty/mis-shaped VD_PEER_HOSTS='$VD_PEER_HOSTS'" >&2; exit 1 ;;
esac

export VD_PEERS VD_PEER_HOSTS
echo "entrypoint: role=$ROLE VD_PEERS=$VD_PEERS VD_PEER_HOSTS=$VD_PEER_HOSTS" >&2
exec "$@"
