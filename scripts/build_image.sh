#!/usr/bin/env bash
# ==============================================================================
# Build `ernie55-dual-spark:v0.3.0` on Node 0 and Sync Over 200GbE to Node 1
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${REPO_ROOT}/.env" && set +a
fi

IMAGE="${IMAGE:-ernie55-dual-spark:v0.3.0}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.32.0}"
NODE1_SSH="${NODE1_SSH:-ubuntu@10.42.0.2}"
SYNC_NODE1=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      [[ $# -ge 2 ]] || { echo "ERROR: --tag requires an image tag" >&2; exit 1; }
      IMAGE="$2"; shift 2 ;;
    --base-image)
      [[ $# -ge 2 ]] || { echo "ERROR: --base-image requires a base image" >&2; exit 1; }
      BASE_IMAGE="$2"; shift 2 ;;
    --no-sync) SYNC_NODE1=0; shift ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--tag IMAGE_TAG] [--base-image BASE] [--no-sync]

Builds ${IMAGE} on Node 0 from ${BASE_IMAGE}, verifies all 9 GB10 (SM121)
patches, and streams the image to Node 1 over the 200GbE SSH link.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "[1/2] Building ${IMAGE} on Node 0 (base: ${BASE_IMAGE})..."
BUILD_ARGS=(
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
)
[[ -n "${HTTP_PROXY:-}" ]] && BUILD_ARGS+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}")
[[ -n "${HTTPS_PROXY:-}" ]] && BUILD_ARGS+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}")
[[ -n "${NO_PROXY:-}" ]] && BUILD_ARGS+=(--build-arg "NO_PROXY=${NO_PROXY}")
[[ -n "${PIP_INDEX_URL:-}" ]] && BUILD_ARGS+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")

docker build "${BUILD_ARGS[@]}" -t "${IMAGE}" "${REPO_ROOT}"
echo "  Built ${IMAGE} on Node 0."

if [[ "${SYNC_NODE1}" -eq 1 ]]; then
  echo "[2/2] Streaming ${IMAGE} over 200GbE fabric to ${NODE1_SSH} ..."
  if command -v zstd >/dev/null 2>&1 && ssh -n -o BatchMode=yes "${NODE1_SSH}" "command -v zstd >/dev/null 2>&1"; then
    docker save "${IMAGE}" | zstd -T0 -1 \
      | ssh -T -o BatchMode=yes -c aes128-gcm@openssh.com -o Compression=no "${NODE1_SSH}" "set -o pipefail; zstd -d | docker load"
  else
    docker save "${IMAGE}" | gzip -1 \
      | ssh -T -o BatchMode=yes -c aes128-gcm@openssh.com -o Compression=no "${NODE1_SSH}" "set -o pipefail; gunzip | docker load"
  fi
  echo "  Loaded ${IMAGE} on ${NODE1_SSH}."
else
  echo "[2/2] Skipped Node 1 sync (--no-sync)."
fi
