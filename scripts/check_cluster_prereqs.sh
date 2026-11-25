#!/usr/bin/env bash
# ==============================================================================
# Pre-Flight Cluster Verification for 2x NVIDIA DGX Spark (GB10 SM121 + RoCEv2)
# Checks GPU capability, unified RAM headroom, ConnectX-7 RoCEv2 GIDs on both
# nodes, SSH, Docker + NVIDIA Container Toolkit, and ERNIE-5.5 weight shards.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${REPO_ROOT}/.env" && set +a
fi

NODE0_IP="${NODE0_IP:-10.42.0.1}"
NODE1_IP="${NODE1_IP:-10.42.0.2}"
NODE1_SSH="${NODE1_SSH:-ubuntu@10.42.0.2}"
FABRIC_IFNAME="${FABRIC_IFNAME:-enp1s0f0np0}"
RDMA_HCAS="${RDMA_HCAS:-rocep1s0f0,roceP2p1s0f0}"
NODE1_FABRIC_IFNAME="${NODE1_FABRIC_IFNAME:-${FABRIC_IFNAME}}"
NODE1_RDMA_HCAS="${NODE1_RDMA_HCAS:-${RDMA_HCAS}}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
MODEL_DIR="${MODEL_DIR:-/data/models/ERNIE-5.5-Flash-Preview}"
NODE1_MODEL_DIR="${NODE1_MODEL_DIR:-${MODEL_DIR}}"
SKIP_WEIGHTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--skip-weights]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

PASS_CNT=0
WARN_CNT=0
FAIL_CNT=0

pass() { echo "  [PASS] $*"; PASS_CNT=$((PASS_CNT + 1)); }
warn() { echo "  [WARN] $*"; WARN_CNT=$((WARN_CNT + 1)); }
fail() { echo "  [FAIL] $*"; FAIL_CNT=$((FAIL_CNT + 1)); }

echo "======================================================================"
echo " ERNIE-5.5-Flash-Preview — Dual DGX Spark Pre-Flight Check"
echo " Node 0: ${NODE0_IP} (${FABRIC_IFNAME}, HCAs=${RDMA_HCAS})"
echo " Node 1: ${NODE1_SSH} / ${NODE1_IP} (${NODE1_FABRIC_IFNAME}, HCAs=${NODE1_RDMA_HCAS})"
echo " RoCEv2 GID Index: ${NCCL_IB_GID_INDEX}"
echo "======================================================================"

echo ""
echo "[1/6] Checking Node 0 & Node 1 SSH Connectivity..."
SSH_OK=0
if ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "echo ok" >/dev/null 2>&1; then
  pass "Passwordless SSH to ${NODE1_SSH} succeeded"
  SSH_OK=1
else
  fail "Cannot SSH to ${NODE1_SSH} in BatchMode (configure ssh-copy-id from Node 0 -> Node 1)"
fi

if ping -c 2 -W 2 "${NODE1_IP}" >/dev/null 2>&1; then
  pass "200GbE fabric ping ${NODE0_IP} -> ${NODE1_IP} reachable"
else
  fail "Cannot reach Node 1 fabric IP ${NODE1_IP} (check cable & IP configuration on ${FABRIC_IFNAME})"
fi

echo ""
echo "[2/6] Checking GPU Architecture (NVIDIA GB10 / Compute Capability 12.1)..."
check_gpu_cmd='nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'
if gpu0="$(eval "${check_gpu_cmd}" 2>/dev/null)" && [[ -n "${gpu0}" ]]; then
  if [[ "${gpu0}" == *"12.1"* ]]; then
    pass "Node 0 GPU: ${gpu0}"
  else
    warn "Node 0 GPU compute capability is not 12.1 (${gpu0}) — patches target SM120/SM121"
  fi
else
  fail "Node 0 nvidia-smi failed"
fi

if [[ "${SSH_OK}" -eq 1 ]] && gpu1="$(ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "${check_gpu_cmd}" 2>/dev/null)" && [[ -n "${gpu1}" ]]; then
  if [[ "${gpu1}" == *"12.1"* ]]; then
    pass "Node 1 GPU: ${gpu1}"
  else
    warn "Node 1 GPU compute capability is not 12.1 (${gpu1})"
  fi
else
  fail "Node 1 nvidia-smi failed via SSH (${NODE1_SSH})"
fi

echo ""
echo "[3/6] Checking Unified Memory Headroom (128 GiB LPDDR5X per node)..."
mem_cmd='awk "/MemTotal/ {t=\$2/1048576} /MemAvailable/ {a=\$2/1048576} END {printf \"%.1f %.1f\n\", t, a}" /proc/meminfo'
if read -r n0_tot n0_avail < <(eval "${mem_cmd}" 2>/dev/null); then
  if awk "BEGIN {exit !(${n0_tot} >= 115.0)}"; then
    pass "Node 0 RAM: total=${n0_tot} GiB, available=${n0_avail} GiB"
  else
    fail "Node 0 total RAM (${n0_tot} GiB) is below 115 GiB"
  fi
  if ! awk "BEGIN {exit !(${n0_avail} >= 105.0)}"; then
    warn "Node 0 MemAvailable (${n0_avail} GiB) < 105 GiB (stop other GPU workloads or pass --drop-page-cache when launching)"
  fi
else
  fail "Could not read Node 0 /proc/meminfo"
fi

if [[ "${SSH_OK}" -eq 1 ]] && read -r n1_tot n1_avail < <(ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "${mem_cmd}" 2>/dev/null); then
  if awk "BEGIN {exit !(${n1_tot} >= 115.0)}"; then
    pass "Node 1 RAM: total=${n1_tot} GiB, available=${n1_avail} GiB"
  else
    fail "Node 1 total RAM (${n1_tot} GiB) is below 115 GiB"
  fi
  if ! awk "BEGIN {exit !(${n1_avail} >= 105.0)}"; then
    warn "Node 1 MemAvailable (${n1_avail} GiB) < 105 GiB"
  fi
else
  fail "Could not query Node 1 /proc/meminfo via SSH (${NODE1_SSH})"
fi

echo ""
echo "[4/6] Checking ConnectX-7 200GbE Interface & RoCEv2 GID Index on Both Nodes..."
if ip -4 addr show dev "${FABRIC_IFNAME}" 2>/dev/null | grep -qF "inet ${NODE0_IP}/"; then
  pass "Node 0 ${FABRIC_IFNAME} has IPv4 ${NODE0_IP}"
else
  fail "Node 0 interface ${FABRIC_IFNAME} does not hold ${NODE0_IP} (if using Port 1, set FABRIC_IFNAME=enp1s0f1np1 and RDMA_HCAS=rocep1s0f1,roceP2p1s0f1 in .env)"
fi

IFS=',' read -r -a hca_arr0 <<< "${RDMA_HCAS}"
for hca in "${hca_arr0[@]}"; do
  gid_type_file="/sys/class/infiniband/${hca}/ports/1/gid_attrs/types/${NCCL_IB_GID_INDEX}"
  if [[ -r "${gid_type_file}" ]]; then
    gtype="$(cat "${gid_type_file}" 2>/dev/null || true)"
    if [[ "${gtype}" == *"RoCE v2"* ]]; then
      pass "Node 0 HCA ${hca} GID[${NCCL_IB_GID_INDEX}] = '${gtype}'"
    else
      warn "Node 0 HCA ${hca} GID[${NCCL_IB_GID_INDEX}] is '${gtype}' (expected 'RoCE v2')"
    fi
  else
    fail "Node 0 RDMA HCA ${hca} missing ${gid_type_file} (check 'rdma link' / RDMA_HCAS in .env)"
  fi
done

if [[ "${SSH_OK}" -eq 1 ]]; then
  if ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "ip -4 addr show dev '${NODE1_FABRIC_IFNAME}'" 2>/dev/null | grep -qF "inet ${NODE1_IP}/"; then
    pass "Node 1 ${NODE1_FABRIC_IFNAME} has IPv4 ${NODE1_IP}"
  else
    fail "Node 1 interface ${NODE1_FABRIC_IFNAME} does not hold ${NODE1_IP} (set NODE1_FABRIC_IFNAME / NODE1_RDMA_HCAS in .env if cabled asymmetrically)"
  fi

  IFS=',' read -r -a hca_arr1 <<< "${NODE1_RDMA_HCAS}"
  for hca in "${hca_arr1[@]}"; do
    gid_path="/sys/class/infiniband/${hca}/ports/1/gid_attrs/types/${NCCL_IB_GID_INDEX}"
    if gtype1="$(ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "cat '${gid_path}'" 2>/dev/null)"; then
      if [[ "${gtype1}" == *"RoCE v2"* ]]; then
        pass "Node 1 HCA ${hca} GID[${NCCL_IB_GID_INDEX}] = '${gtype1}'"
      else
        warn "Node 1 HCA ${hca} GID[${NCCL_IB_GID_INDEX}] is '${gtype1}' (expected 'RoCE v2')"
      fi
    else
      fail "Node 1 RDMA HCA ${hca} missing ${gid_path}"
    fi
  done
else
  fail "Skipped Node 1 RDMA verification because SSH to ${NODE1_SSH} failed"
fi

echo ""
echo "[5/6] Checking Docker & NVIDIA Container Runtime..."
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q "nvidia"; then
  pass "Node 0 Docker has nvidia runtime registered"
else
  fail "Node 0 Docker missing nvidia runtime (install nvidia-container-toolkit)"
fi
if [[ "${SSH_OK}" -eq 1 ]] && ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "docker info --format '{{json .Runtimes}}'" 2>/dev/null | grep -q "nvidia"; then
  pass "Node 1 Docker has nvidia runtime registered"
else
  fail "Node 1 Docker missing nvidia runtime (or SSH unreachable)"
fi

echo ""
echo "[6/6] Checking Model Weights (128 safetensors shards + flashindexer/)..."
if [[ "${SKIP_WEIGHTS}" -eq 1 ]]; then
  warn "Skipped weight check (--skip-weights)"
else
  check_w='d="$1"; if [[ ! -d "$d" ]]; then echo "0 0 0"; exit 0; fi; c=$(find "$d" -maxdepth 1 -type f -name "model-*-of-00128.safetensors" 2>/dev/null | wc -l | tr -d " "); fi=$(find "$d/flashindexer" -maxdepth 1 -type f -name "*.safetensors" 2>/dev/null | wc -l | tr -d " "); cfg=0; [[ -f "$d/config.json" ]] && cfg=1; echo "$c $fi $cfg"'
  read -r n0_shards n0_fi n0_cfg < <(bash -c "${check_w}" _ "${MODEL_DIR}")
  if [[ "${n0_shards}" -eq 128 && "${n0_fi}" -ge 1 && "${n0_cfg}" -eq 1 ]]; then
    pass "Node 0 weights complete at ${MODEL_DIR} (128/128 shards + flashindexer + config.json)"
  else
    fail "Node 0 weights incomplete at ${MODEL_DIR} (shards=${n0_shards}/128, flashindexer_safetensors=${n0_fi}, config=${n0_cfg}); run ./scripts/prepare_weights.sh"
  fi

  if [[ "${SSH_OK}" -eq 1 ]] && read -r n1_shards n1_fi n1_cfg < <(ssh -n -o BatchMode=yes -o ConnectTimeout=6 "${NODE1_SSH}" "bash -c '${check_w}' _ '${NODE1_MODEL_DIR}'" 2>/dev/null); then
    if [[ "${n1_shards}" -eq 128 && "${n1_fi}" -ge 1 && "${n1_cfg}" -eq 1 ]]; then
      pass "Node 1 weights complete at ${NODE1_MODEL_DIR} (128/128 shards + flashindexer + config.json)"
    else
      fail "Node 1 weights incomplete at ${NODE1_MODEL_DIR} (shards=${n1_shards}/128, flashindexer_safetensors=${n1_fi}, config=${n1_cfg}); run ./scripts/prepare_weights.sh --sync-only"
    fi
  else
    fail "Could not verify weights on ${NODE1_SSH}:${NODE1_MODEL_DIR}"
  fi
fi

echo ""
echo "----------------------------------------------------------------------"
echo " Summary: ${PASS_CNT} passed, ${WARN_CNT} warnings, ${FAIL_CNT} failed"
echo "----------------------------------------------------------------------"
[[ "${FAIL_CNT}" -eq 0 ]]
