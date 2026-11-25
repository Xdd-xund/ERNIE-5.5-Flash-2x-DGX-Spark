# syntax=docker/dockerfile:1
# ==============================================================================
# Reproducible ARM64 GB10 (SM121) Runtime Image for `baidu/ERNIE-5.5-Flash-Preview`
# Target Hardware: 2x NVIDIA DGX Spark (Grace-Blackwell GB10, 128 GiB Unified RAM,
#                  Dual ConnectX-7 200GbE RoCEv2, Tensor Parallelism TP=2)
#
# Build:
#   docker build -t ernie55-dual-spark:v0.3.0 .
# ==============================================================================

ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${BASE_IMAGE}

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY=""
ARG PIP_INDEX_URL=""
ARG VLLM_ERNIE_REPO=https://github.com/baidu/vLLM-ERNIE.git
ARG VLLM_ERNIE_COMMIT=ba322a5634f91d3e6f96ae711df72ddaf3c38b6e

USER root
WORKDIR /vllm-workspace

# 1. Ensure ConnectX-7 RoCEv2 user-space verbs providers and git are installed
#    (export both lowercase & uppercase proxy vars across subcommands without baking into image ENV)
RUN export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}" no_proxy="${NO_PROXY}" \
           HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" NO_PROXY="${NO_PROXY}" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        ibverbs-providers \
        libibverbs1 \
        libibverbs-dev \
        rdma-core \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Pin CUDA 13.0 PyTorch + NCCL 2.30.7, install SM120/SM121 `b12x` + FlashInfer + TileLang + CuTe DSL,
#    and remove pre-compiled SM90/SM100 `flashinfer-cubin`/`flashinfer-jit-cache` so SM121 JIT compiles cleanly.
RUN export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}" no_proxy="${NO_PROXY}" \
           HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" NO_PROXY="${NO_PROXY}" \
    && printf "nvidia-nccl-cu13==2.30.7\ntorch==2.13.0+cu130\ntorchaudio==2.11.0+cu130\ntorchvision==0.28.0+cu130\n" > /etc/uv-overrides.txt \
    && UV_OVERRIDE=/etc/uv-overrides.txt uv pip install --system --no-cache \
         --index-strategy unsafe-best-match \
         --extra-index-url https://download.pytorch.org/whl/cu130 \
         ${PIP_INDEX_URL:+--index-url "${PIP_INDEX_URL}"} \
         "vllm[b12x]==0.32.0" \
         "b12x==1.5.1" \
         "flashinfer-python==0.7.0.post1" \
         "tilelang==0.1.12" \
         "nvidia-cutlass-dsl==4.8.0" \
         "fastsafetensors==0.4.0" \
    && (uv pip uninstall --system flashinfer-cubin flashinfer-jit-cache || true)

# 3. Fetch pinned `baidu/vLLM-ERNIE` plugin commit and install in editable source mode
RUN export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}" no_proxy="${NO_PROXY}" \
           HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" NO_PROXY="${NO_PROXY}" \
    && git init /opt/vLLM-ERNIE \
    && git -C /opt/vLLM-ERNIE remote add origin "${VLLM_ERNIE_REPO}" \
    && git -C /opt/vLLM-ERNIE fetch --depth 1 origin "${VLLM_ERNIE_COMMIT}" \
    && git -C /opt/vLLM-ERNIE checkout FETCH_HEAD \
    && UV_OVERRIDE=/etc/uv-overrides.txt uv pip install --system --no-cache --no-build-isolation --no-deps -e /opt/vLLM-ERNIE

# 4. Apply and byte-compile verify all 9 GB10 (SM121) adaptation & v0.3.0 speed patches
COPY patches/patch_gb10_ernie55.py /opt/vLLM-ERNIE/patch_gb10_ernie55.py
RUN python3 /opt/vLLM-ERNIE/patch_gb10_ernie55.py

LABEL org.opencontainers.image.title="ernie55-dual-spark" \
      org.opencontainers.image.version="0.3.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.ernie.dual_spark.version="0.3.0" \
      ai.ernie.vllm_version="0.32.0" \
      ai.ernie.vllm_ernie_commit="${VLLM_ERNIE_COMMIT}" \
      ai.ernie.arch="arm64-gb10-sm121" \
      ai.ernie.patches="sm121-hca-dsa-sink vha-bf16-prefold gated-deepgemm-tp2 b12x-tinydecode-situ flashindexer-zerocopy-pad16 dspark-fullvocab-fp8-markov-multimax rocev2-ll-simple"

ENV VLLM_IMAGE_TAG=ernie55-dual-spark:v0.3.0 \
    MBX_LOAD_DROP_CACHE=1

ENTRYPOINT ["vllm", "serve"]
