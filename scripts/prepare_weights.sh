#!/usr/bin/env bash
# ==============================================================================
# Download & Sync `baidu/ERNIE-5.5-Flash-Preview` (~154.5 GiB, 128 shards)
# Downloads weights onto Node 0 via HuggingFace CLI (pinned revision) or
# ModelScope, and mirrors across the 200GbE ConnectX-7 fabric to Node 1.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${REPO_ROOT}/.env" && set +a
fi

MODEL_REPO="${MODEL_REPO:-baidu/ERNIE-5.5-Flash-Preview}"
MODEL_REVISION="${MODEL_REVISION:-ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe}"
MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-}"
MODEL_DIR="${MODEL_DIR:-/data/models/ERNIE-5.5-Flash-Preview}"
NODE1_MODEL_DIR_ENV="${NODE1_MODEL_DIR:-}"
NODE1_SSH="${NODE1_SSH:-ubuntu@10.42.0.2}"
SOURCE_KIND="hf"
SYNC_NODE1=1
SYNC_ONLY=0
CLI_NODE1_MODEL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modelscope) SOURCE_KIND="modelscope"; shift ;;
    --no-sync) SYNC_NODE1=0; shift ;;
    --sync-only) SYNC_ONLY=1; shift ;;
    --model-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --model-dir requires a path" >&2; exit 1; }
      MODEL_DIR="$2"; shift 2 ;;
    --node1-model-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --node1-model-dir requires a path" >&2; exit 1; }
      CLI_NODE1_MODEL_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--modelscope] [--no-sync] [--sync-only] [--model-dir PATH] [--node1-model-dir PATH]

Options:
  --modelscope          Download from ModelScope instead of HuggingFace Hub
  --no-sync             Only download onto Node 0 (skip rsync to Node 1)
  --sync-only           Skip download; only rsync existing MODEL_DIR from Node 0 -> Node 1
  --model-dir PATH      Override Node 0 destination path (default: ${MODEL_DIR})
  --node1-model-dir P   Override Node 1 destination path (defaults to NODE1_MODEL_DIR or MODEL_DIR)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${CLI_NODE1_MODEL_DIR}" ]]; then
  NODE1_MODEL_DIR="${CLI_NODE1_MODEL_DIR}"
elif [[ -n "${NODE1_MODEL_DIR_ENV}" ]]; then
  NODE1_MODEL_DIR="${NODE1_MODEL_DIR_ENV}"
else
  NODE1_MODEL_DIR="${MODEL_DIR}"
fi

export PATH="${HOME}/.local/bin:${PATH}"

install_user_cli() {
  if command -v uv >/dev/null 2>&1; then
    uv tool install --upgrade "$1"
  else
    python3 -m pip install --user --break-system-packages -U "$1" \
      || python3 -m pip install --user -U "$1"
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
}

mkdir -p "${MODEL_DIR}"

if [[ "${SYNC_ONLY}" -eq 0 ]]; then
  if [[ "${SOURCE_KIND}" == "modelscope" ]]; then
    echo "[1/3] Downloading ${MODEL_REPO} from ModelScope -> ${MODEL_DIR}"
    if ! command -v modelscope >/dev/null 2>&1; then
      echo "Installing modelscope CLI in user environment..."
      install_user_cli "modelscope"
    fi
    MS_ARGS=(download --model "${MODEL_REPO}" --local_dir "${MODEL_DIR}")
    [[ -n "${MODELSCOPE_REVISION}" ]] && MS_ARGS+=(--revision "${MODELSCOPE_REVISION}")
    modelscope "${MS_ARGS[@]}"
  else
    echo "[1/3] Downloading ${MODEL_REPO} (HF revision: ${MODEL_REVISION}) -> ${MODEL_DIR}"
    if command -v hf >/dev/null 2>&1; then
      HF_XET_HIGH_PERFORMANCE=1 hf download "${MODEL_REPO}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_DIR}"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "${MODEL_REPO}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_DIR}"
    else
      echo "Installing huggingface_hub[cli,hf_xet] in user environment..."
      install_user_cli "huggingface_hub[cli,hf_xet]"
      HF_XET_HIGH_PERFORMANCE=1 hf download "${MODEL_REPO}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_DIR}"
    fi
  fi
fi

echo "[2/3] Verifying Node 0 weight completeness..."
SHARD_COUNT="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'model-*-of-00128.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
FI_COUNT="$(find "${MODEL_DIR}/flashindexer" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${SHARD_COUNT}" -ne 128 ]]; then
  echo "ERROR: Expected 128 safetensors shards in ${MODEL_DIR}, found ${SHARD_COUNT}" >&2
  exit 1
fi
if [[ "${FI_COUNT}" -lt 1 || ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "ERROR: Missing config.json or flashindexer/*.safetensors in ${MODEL_DIR}" >&2
  exit 1
fi
echo "  Verified 128/128 safetensors shards + flashindexer/ + config.json in ${MODEL_DIR}"

if [[ "${SYNC_NODE1}" -eq 1 ]]; then
  echo "[3/3] Syncing weights over 200GbE fabric to ${NODE1_SSH}:${NODE1_MODEL_DIR} ..."
  ssh -n -o BatchMode=yes "${NODE1_SSH}" "mkdir -p '${NODE1_MODEL_DIR}'"
  rsync -ah --info=progress2 --partial \
    --exclude=".cache/" \
    --exclude="._____temp/" \
    -e "ssh -T -o BatchMode=yes -c aes128-gcm@openssh.com -o Compression=no" \
    "${MODEL_DIR}/" "${NODE1_SSH}:${NODE1_MODEL_DIR}/"
  echo "  Sync complete on ${NODE1_SSH}:${NODE1_MODEL_DIR}"
else
  echo "[3/3] Skipped Node 1 sync (--no-sync)."
fi
