# Third-Party Notices

This repository (`ERNIE-5.5-Flash-2x-DGX-Spark`) provides build recipes, cluster orchestration scripts, and build-time adaptation patches under the **Apache License 2.0** (see `LICENSE`).

It does **not** redistribute model weights, proprietary binaries, or full copies of upstream repositories. When building the container image via `Dockerfile` or downloading weights via `scripts/prepare_weights.sh`, the following third-party open-source and vendor components are fetched directly from their official upstream sources and remain subject to their respective licenses:

---

## 1. Baidu `ERNIE-5.5-Flash-Preview` & `vLLM-ERNIE`
- **Model Repository**: https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview (`ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe`)
- **Plugin Repository**: https://github.com/baidu/vLLM-ERNIE (`ba322a5634f91d3e6f96ae711df72ddaf3c38b6e`)
- **License**: Apache License 2.0
- **Notice**: `patches/patch_gb10_ernie55.py` applies build-time in-place adaptations to `vllm_ernie` (`attention.py`, `flashindexer.py`, `model.py`, `dspark.py`, `patches.py`) to enable `SM121` (`GB10`) HCA + MLA-DSA sparse attention, `TP=2` `GatedDeepGemmLinear`, `VHAPostMix` `bfloat16` pre-folding, zero-copy `FlashIndexer` 16-head padding, and full-vocabulary fused FP8 `DSpark` Markov + Multimax speculative decoding.

## 2. vLLM (`vllm/vllm-openai:v0.32.0`)
- **Repository**: https://github.com/vllm-project/vllm (`5df63a7203cdb9328d824a27b682e77c7b9928be`)
- **Container Image**: `docker.io/vllm/vllm-openai:v0.32.0`
- **License**: Apache License 2.0
- **Notice**: `patches/patch_gb10_ernie55.py` applies build-time in-place adaptations to `vllm` (`weight_utils.py`, `shm_broadcast.py`, `config/speculative.py`, `v1/worker/gpu/spec_decode/dspark/speculator.py`, and `sm121_*.cuh` alias headers for bundled `DeepGEMM`).

## 3. `b12x` (Blackwell SM120/SM121 MoE Kernels)
- **Package**: https://pypi.org/project/b12x/ (`b12x==1.5.1`)
- **License**: Apache License 2.0 (see package metadata)
- **Notice**: `patches/patch_gb10_ernie55.py` patches `b12x` (`situ.py`, `tiny_decode.py`, `fused_moe/_impl.py`) at image build time to forward `*args, **kwargs` in `MoEDynamicKernelSitu` and unlock exact `SiTU` epilogue execution in `MoETinyDecodeKernelBackend` for `1 <= M <= 3`.

## 4. FlashInfer, DeepGEMM, TileLang, and Triton
- **FlashInfer** (`flashinfer-python==0.7.0.post1`): https://github.com/flashinfer-ai/flashinfer — Apache License 2.0
- **DeepGEMM** (bundled in `vLLM`): https://github.com/deepseek-ai/DeepGEMM — MIT License
- **TileLang** (`tilelang==0.1.12`): https://github.com/tile-ai/tilelang — MIT License
- **Triton** (`triton`): https://github.com/triton-lang/triton — MIT License

## 5. NVIDIA CUDA, NCCL, and CUTLASS DSL
- **CUDA Runtime & Toolkit** (inside base container `vllm/vllm-openai:v0.32.0`): Subject to the [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license) and NVIDIA CUDA EULA.
- **NCCL** (`nvidia-nccl-cu13==2.30.7`): https://github.com/NVIDIA/nccl — BSD-3-Clause License
- **NVIDIA CUTLASS DSL** (`nvidia-cutlass-dsl==4.8.0`): https://pypi.org/project/nvidia-cutlass-dsl/ — Subject to NVIDIA CUTLASS / CuTe DSL license terms.
