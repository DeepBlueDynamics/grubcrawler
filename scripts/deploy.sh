#!/usr/bin/env bash
# ============================================================
# deploy.sh — grub-crawl deployment (local / mesh / cloudrun)
#
# Lives in scripts/ but resolves the project root itself, so it can be
# invoked from anywhere:
#   ./scripts/deploy.sh local                    # single node
#   ./scripts/deploy.sh mesh                     # 2-node mesh
#   ./scripts/deploy.sh cloudrun [tag]           # Cloud Run
#   ./scripts/deploy.sh cloudrun v1.0.0 --mesh-peer http://localhost:6792 --mesh-secret mykey
# ============================================================
set -euo pipefail

# Always run from the project root (parent of this script) so docker build
# context and docker-compose files resolve regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TARGET="${1:-local}"
TAG="${2:-latest}"
MESH_PEER=""
MESH_SECRET=""

# Consume TARGET and TAG positionals. Tolerate a single argument (the original
# `shift 2` failed when only TARGET was given, leaving "local" to fall through
# to the flag parser as "Unknown flag: local"). Shift at most $# args.
shift "$(( $# < 2 ? $# : 2 ))" 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mesh-peer)  MESH_PEER="$2";  shift 2 ;;
        --mesh-secret) MESH_SECRET="$2"; shift 2 ;;
        --rebuild)     REBUILD="--no-cache"; shift ;;
        *)             echo "Unknown flag: $1"; exit 1 ;;
    esac
done

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-gnosis-459403}"
REGION="us-central1"
SERVICE_NAME="grub-crawl"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
REBUILD="${REBUILD:-}"

echo "==> Deploying ${SERVICE_NAME} to ${TARGET}"

# ---------------------------------------------------------------------------
case "$TARGET" in
  local)
    echo "==> Building image..."
    docker build ${REBUILD} -t "${SERVICE_NAME}:${TAG}" .
    echo "==> Starting single node..."
    docker-compose -f docker-compose.mesh.yml down 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    docker-compose up -d
    echo ""
    echo "==> Service deployed locally"
    echo "    API:      http://localhost:6792"
    echo "    Health:   http://localhost:6792/health"
    echo "    Site:     http://localhost:6792/site"
    echo ""
    docker-compose logs --tail=10
    ;;

  mesh)
    echo "==> Building image..."
    docker build ${REBUILD} -t "${SERVICE_NAME}:${TAG}" .
    echo "==> Starting 2-node mesh..."
    docker-compose down 2>/dev/null || true
    docker-compose -f docker-compose.mesh.yml down 2>/dev/null || true
    docker-compose -f docker-compose.mesh.yml up -d --build
    echo ""
    echo "==> Mesh deployed (2 nodes)"
    echo "    Node A:   http://localhost:6792  (local)"
    echo "    Node B:   http://localhost:6793  (cloud)"
    echo "    Peers A:  http://localhost:6792/mesh/peers"
    echo "    Peers B:  http://localhost:6793/mesh/peers"
    echo "    Site:     http://localhost:6792/site"
    echo ""
    sleep 5
    docker-compose -f docker-compose.mesh.yml logs --tail=15
    ;;

  cloudrun)
    echo "==> Setting project to ${PROJECT_ID}"
    gcloud config set project "${PROJECT_ID}"
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

    FULL_IMAGE="${IMAGE}:${TAG}"
    echo "==> Building and pushing image..."
    docker build ${REBUILD} -t "${FULL_IMAGE}" .
    docker push "${FULL_IMAGE}"

    # Base env vars
    ENV_VARS="RUNNING_IN_CLOUD=true"
    ENV_VARS="${ENV_VARS},GCS_BUCKET_NAME=grub-crawl-storage-prod"
    ENV_VARS="${ENV_VARS},GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"

    # Mesh env vars
    if [[ -n "$MESH_PEER" ]]; then
        if [[ -z "$MESH_SECRET" ]]; then
            MESH_SECRET=$(openssl rand -hex 12)
            echo "==> Generated mesh secret: ${MESH_SECRET}"
        fi
        ENV_VARS="${ENV_VARS},MESH_ENABLED=true"
        ENV_VARS="${ENV_VARS},MESH_NODE_NAME=cloud"
        ENV_VARS="${ENV_VARS},MESH_SECRET=${MESH_SECRET}"
        ENV_VARS="${ENV_VARS},MESH_PEERS=${MESH_PEER}"
        ENV_VARS="${ENV_VARS},MESH_PREFER_LOCAL=false"
    fi

    echo "==> Deploying to Cloud Run..."
    gcloud run deploy "${SERVICE_NAME}" \
        --image "${FULL_IMAGE}" \
        --platform managed \
        --region "${REGION}" \
        --allow-unauthenticated \
        --port 6792 \
        --memory 1Gi \
        --cpu 1 \
        --max-instances 10 \
        --timeout 300 \
        --concurrency 100 \
        --set-env-vars "${ENV_VARS}"

    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format "value(status.url)")

    # Update advertise URL if mesh
    if [[ -n "$MESH_PEER" ]]; then
        gcloud run services update "${SERVICE_NAME}" \
            --region "${REGION}" \
            --update-env-vars "MESH_ADVERTISE_URL=${SERVICE_URL}"
    fi

    echo ""
    echo "==> Service deployed to Cloud Run"
    echo "    URL:      ${SERVICE_URL}"
    echo "    Health:   ${SERVICE_URL}/health"
    echo "    Site:     ${SERVICE_URL}/site"

    if [[ -n "$MESH_PEER" ]]; then
        echo ""
        echo "==> Mesh enabled"
        echo "    Peers:    ${SERVICE_URL}/mesh/peers"
        echo "    To connect local node:"
        echo "    MESH_ENABLED=true MESH_SECRET=${MESH_SECRET} MESH_PEERS=${SERVICE_URL} \\"
        echo "      MESH_ADVERTISE_URL=http://your-local-ip:6792 \\"
        echo "      uvicorn app.main:app --port 6792"
    fi
    ;;

  *)
    echo "Usage: $0 {local|mesh|cloudrun} [tag] [--mesh-peer URL] [--mesh-secret SECRET]"
    exit 1
    ;;
esac

echo ""
echo "==> Deployment completed."