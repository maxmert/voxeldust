#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="voxeldust"
NAMESPACE="voxeldust"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-up}" in
  up)
    echo "==> Creating k3d cluster..."
    k3d cluster create "$CLUSTER_NAME" \
      -p "7777:30777@server:0" \
      -p "10000-10099:10000-10099/tcp@server:0" \
      -p "10000-10099:10000-10099/udp@server:0" \
      --agents 0 \
      --wait 2>/dev/null || {
        echo "Cluster already exists or failed to create. Trying to start..."
        k3d cluster start "$CLUSTER_NAME" 2>/dev/null || true
      }

    echo "==> Building Docker images..."
    docker build --target orchestrator -t voxeldust-orchestrator:latest "$SCRIPT_DIR"
    docker build --target gateway -t voxeldust-gateway:latest "$SCRIPT_DIR"
    docker build --target stub-shard -t voxeldust-stub-shard:latest "$SCRIPT_DIR"
    docker build --target system-shard -t voxeldust-system-shard:latest "$SCRIPT_DIR"
    docker build --target ship-shard -t voxeldust-ship-shard:latest "$SCRIPT_DIR"
    docker build --target planet-shard -t voxeldust-planet-shard:latest "$SCRIPT_DIR"

    echo "==> Importing images into k3d..."
    k3d image import \
      voxeldust-orchestrator:latest \
      voxeldust-gateway:latest \
      voxeldust-stub-shard:latest \
      voxeldust-system-shard:latest \
      voxeldust-ship-shard:latest \
      voxeldust-planet-shard:latest \
      -c "$CLUSTER_NAME"

    echo "==> Applying K8s manifests..."
    kubectl apply -f "$SCRIPT_DIR/k8s/namespace.yaml"
    kubectl apply -f "$SCRIPT_DIR/k8s/rbac.yaml"
    kubectl apply -f "$SCRIPT_DIR/k8s/orchestrator.yaml"
    kubectl apply -f "$SCRIPT_DIR/k8s/gateway.yaml"

    echo "==> Waiting for deployments..."
    kubectl -n "$NAMESPACE" rollout status deployment/orchestrator --timeout=120s
    kubectl -n "$NAMESPACE" rollout status deployment/gateway --timeout=120s

    echo ""
    echo "Voxeldust cluster is ready!"
    echo ""
    echo "  Gateway:          localhost:7777 (TCP, for game clients)"
    echo "  Orchestrator API: kubectl -n $NAMESPACE port-forward svc/orchestrator 8080:8080"
    echo "  Shard pods:       kubectl -n $NAMESPACE get pods -l app=voxeldust-shard"
    echo "  Logs:             kubectl -n $NAMESPACE logs -f deployment/orchestrator"
    echo ""
    echo "  Test provisioning: curl http://localhost:8080/system/42  (after port-forward)"
    echo "  Tear down:         ./dev-cluster.sh down"
    ;;

  down)
    echo "==> Deleting k3d cluster..."
    k3d cluster delete "$CLUSTER_NAME"
    echo "Cluster deleted."
    ;;

  rebuild)
    echo "==> Rebuilding Docker images..."
    docker build --target orchestrator -t voxeldust-orchestrator:latest "$SCRIPT_DIR"
    docker build --target gateway -t voxeldust-gateway:latest "$SCRIPT_DIR"
    docker build --target stub-shard -t voxeldust-stub-shard:latest "$SCRIPT_DIR"
    docker build --target system-shard -t voxeldust-system-shard:latest "$SCRIPT_DIR"
    docker build --target ship-shard -t voxeldust-ship-shard:latest "$SCRIPT_DIR"
    docker build --target planet-shard -t voxeldust-planet-shard:latest "$SCRIPT_DIR"

    echo "==> Importing images into k3d..."
    k3d image import \
      voxeldust-orchestrator:latest \
      voxeldust-gateway:latest \
      voxeldust-stub-shard:latest \
      voxeldust-system-shard:latest \
      voxeldust-ship-shard:latest \
      voxeldust-planet-shard:latest \
      -c "$CLUSTER_NAME"

    echo "==> Restarting deployments..."
    kubectl -n "$NAMESPACE" rollout restart deployment/orchestrator
    kubectl -n "$NAMESPACE" rollout restart deployment/gateway

    echo "==> Waiting for rollout..."
    kubectl -n "$NAMESPACE" rollout status deployment/orchestrator --timeout=120s
    kubectl -n "$NAMESPACE" rollout status deployment/gateway --timeout=120s

    echo "Rebuild complete."
    ;;

  status)
    echo "==> Cluster status:"
    kubectl -n "$NAMESPACE" get pods -o wide
    echo ""
    echo "==> Shard pods:"
    kubectl -n "$NAMESPACE" get pods -l app=voxeldust-shard -o wide 2>/dev/null || echo "  (none)"
    ;;

  logs)
    COMPONENT="${2:-orchestrator}"
    kubectl -n "$NAMESPACE" logs -f "deployment/$COMPONENT"
    ;;

  *)
    echo "Usage: $0 {up|down|rebuild|status|logs [component]}"
    exit 1
    ;;
esac
