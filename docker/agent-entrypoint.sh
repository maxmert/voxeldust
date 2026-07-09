#!/bin/sh
# S5a agent entrypoint: getent-resolve the gateway headless-Service pod DNS to an IPv4 (the client's --gateway
# is a raw SocketAddr — no DNS resolver, client.rs), export it, then exec the selected scenario. Reuses the
# resolve() idiom from docker/entrypoint.sh verbatim. VD_SCENARIO selects the baked scenario-<name>.sh script.
set -eu

NS="${VD_K8S_NAMESPACE:-voxeldust}"
DOM="svc.cluster.local"
MESH_PORT="${VD_MESH_PORT:-9000}"

resolve() {
  i=0
  while [ "$i" -lt 90 ]; do
    ip="$(getent ahostsv4 "$1" | awk '{ print $1; exit }')" || true
    if [ -n "${ip:-}" ]; then
      echo "$ip"
      return 0
    fi
    echo "[agent] waiting for DNS $1 (attempt $i)" >&2
    sleep 1
    i=$((i + 1))
  done
  echo "[agent] FATAL $1 never resolved after 90s" >&2
  return 1
}

GW="$(resolve "vd-gateway-0.vd-gateway.$NS.$DOM")" || exit 10
export VD_GATEWAY_ADDR="$GW:$MESH_PORT"
SCENARIO="${VD_SCENARIO:-boundary}"
echo "[agent] gateway=$VD_GATEWAY_ADDR scenario=$SCENARIO" >&2
exec "scenario-$SCENARIO.sh"
