# Architecture, SM121 Kernel Adaptations, and Speed Optimizations

This document explains **why** `baidu/ERNIE-5.5-Flash-Preview` fails out-of-the-box on NVIDIA DGX Spark (`Grace-Blackwell GB10`, `DeviceCapability(12, 1)`) and **how** each of the 9 surgical patches in [`patches/patch_gb10_ernie55.py`](../patches/patch_gb10_ernie55.py) + the RoCEv2 collective profile in [`scripts/launch_cluster.py`](../scripts/launch_cluster.py) unlock full speed and `131K` context capacity across 2× DGX Spark.

---

## 1. Model & Hardware Profile

### `baidu/ERNIE-5.5-Flash-Preview`
- **Total / Active Parameters**: `280B` total / `~14B` active per token
- **Vocabulary**: `201,216` tokens + piecewise-quadratic `Multimax` logit calibration
- **Hybrid Attention Stack (`43` target layers + `3` `DSpark` speculative draft layers = `46` layers)**:
  - **`37` Target + `3` Draft `HCA` (Heavily Compressed Attention) layers**: `584 B/tok` `fp8_ds_mla` KV layout + per-head `attn_sink` + `VHAPostMix` low-rank cross-head mixing (`U, V`).
  - **`6` Target `MLA-DSA` (Dynamic Sparse Attention) layers**: `656 B/tok` `fp8_ds_mla` KV layout (`index_topk = 2048` + `SWA = 128` = `2176` sparse tokens) + `FlashIndexer` (`8` top-k heads selected per token via MXFP4/FP8 MQA logits) + per-head `attn_sink`.
- **MoE FFN**: `W4A8` MXFP4 weights + FP8 activations with custom `SiTU` gated activation (`beta = 4.0`, `lbeta = 25.0`).
- **Speculative Decoding (`DSpark`)**: Bundled 3-layer HCA+MoE draft network + `DSparkMarkovHead` (`markov_w1: [201216, 512]`, `markov_w2: [201216, 512]`) proposing `k = 7` draft tokens (`8` tokens verified per target forward step).

### Dual NVIDIA DGX Spark (`TP = 2`)
- **Compute**: 2× Grace-Blackwell `GB10` (`20` ARMv9.2 cores + `SM121` Blackwell GPU per node)
- **Memory**: 2× `128 GiB` coherent unified LPDDR5X (`~273 GB/s` per node, `~546 GB/s` aggregate across `TP = 2`)
  - Checkpoint footprint: `154.5 GiB` total $\rightarrow$ **`~77.3 GiB` per node**
  - Unified-memory constraint: Linux OS page cache shares the exact same `128 GiB` physical pool as CUDA allocations!
- **Interconnect**: Dual-port NVIDIA ConnectX-7 `200 GbE` RoCEv2 (`ibverbs` / RDMA)

---

## 2. The 9 Surgical Patches (`patches/patch_gb10_ernie55.py`)

### Patch 1 — `DeepGEMM` `SM121` Header Aliases (`sm121_*.cuh` $\rightarrow$ `sm120_*.cuh`)
- **Problem**: `vLLM 0.32.0`'s bundled `DeepGEMM` JIT generates `#include <deep_gemm/impls/sm121_*.cuh>` when `compute_cap == 12.1`, whereas upstream only ships `sm120_*.cuh` headers for consumer/workstation Blackwell (`SM120`).
- **Fix**: Generates idempotent `sm121_<kernel>.cuh` alias headers forwarding to `sm120_<kernel>.cuh` across all 11 Blackwell DeepGEMM kernels (`fp8_mqa_logits`, `fp8_paged_mqa_logits`, `fp8_fp4_gemm_1d1d`, `tf32_hc_prenorm_gemm`, etc.).

### Patch 2 — Unified-Memory Safetensors Page-Cache Drop (`POSIX_FADV_DONTNEED`)
- **Problem**: Reading `77.3 GiB` of `.safetensors` shards per node populates `~77 GiB` of Linux OS buffer/page cache inside the `128 GiB` unified LPDDR5X pool, triggering host OOM during weight loading + KV cache allocation.
- **Fix**: Injects `_mbx_fadvise_dontneed(st_file)` (`os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)`) immediately after each shard finishes streaming in `vllm/model_executor/model_loader/weight_utils.py` (`MBX_LOAD_DROP_CACHE=1`), keeping `buff/cache` under `1.5–4.5 GiB` without requiring root `drop_caches`.

### Patch 3 — Grace ARM64 IPC Spin-Wait Reduction (`shm_broadcast.py`)
- **Problem**: `vLLM`'s `MessageQueue` defaults to `busy_loop_s = 1` second of tight CPU spinning before sleeping, burning Grace P-core thermal/power budget when idle.
- **Fix**: Lowers `busy_loop_s` from `1` to `0.002` (`2 ms`), preserving sub-microsecond burst wakeups while eliminating idle CPU spin.

### Patch 4 — Native `SM120`/`SM121` HCA + Sink-Capable `MLA-DSA` Sparse Attention
- **Problem**: Upstream `vLLM-ERNIE` hardcodes `FlashMLA` (`Ernie5_5HCAFlashMLAAttention` and `Ernie5_5FlashMLASparseBackend`) whenever `--kv-cache-dtype fp8` is enabled. `FlashMLA` targets Hopper/B200 (`SM90`/`SM100`) WGMMA/TCGEN05 instructions and cannot execute on `SM121` (`GB10`).
- **Fix**:
  - Adds `Ernie5_5HCAFlashInferSM120Attention(Ernie5_5HCAAttentionMixin, DeepseekV4FlashInferSM120Attention)` for all `40` HCA layers (`37` target + `3` draft) when `cap.major == 12`.
  - Adds `Ernie5_5FlashInferMLASparseSM120Backend` and `Ernie5_5FlashInferMLASparseSM120Impl` for all `6` `MLA-DSA` layers, forwarding per-head `sinks=[self.sinks]` into `flashinfer_trtllm_batch_decode_with_kv_cache_mla`.
  - Note: Requires uninstalling stale pre-built `flashinfer-cubin` / `flashinfer-jit-cache` wheels so `flashinfer-python==0.7.0.post1` JIT-compiles clean `SM121` kernels into the persistent host cache.

### Patch 5 — `VHAPostMix` Pre-Folded Contiguous `U^T` + Native `bfloat16` `AllReduce`
- **Problem**: Every attention layer (`46` per step) runs `VHAPostMix` (`o + V @ all_reduce(U^T @ o)`). Upstream `vLLM-ERNIE` folds `I + V U^T` in `bfloat16` when `TP=1`, but on `TP>1` materialized non-contiguous `self.U.t()` every layer and upcast the `AllReduce` tensor to `float32` — accounting for **68% of cross-node RoCEv2 bytes** per decode step.
- **Fix**: Pre-folds contiguous `Ut` buffers at weight load time and runs the `AllReduce` in native `bfloat16` (matching `TP=1` precision; cosine similarity `0.9999994` vs `fp32`), cutting cross-node VHA traffic in half.

### Patch 6 — Unlocking `GatedDeepGemmLinear` for `TP = 2`
- **Problem**: Upstream `GatedDeepGemmLinear.wrap()` had a `get_tensor_model_parallel_world_size() != 1` guard, forcing all `46` HCA + DSA `wo_b` output projections onto unfused `torch.sigmoid(gate) * x` $\rightarrow$ separate UE8M0 quantization $\rightarrow$ linear GEMM.
- **Fix**: Validates `local_in % 512 == 0` on `RowParallelLinear` (`input_size_per_partition`) and executes `tensor_model_parallel_all_reduce(out)` when `reduce_results=True` (`max_abs_diff = 0.0`, bit-exact).

### Patch 7 — `b12x` `MoEDynamicKernelSitu` Fix + `TinyDecode-SITU` for `1 <= M <= 3`
- **Problem**:
  1. `b12x==1.5.1`'s `MoEDynamicKernelSitu.__init__` dropped `*args, **kwargs`, breaking `w4a8_n64_repacked` forwarding.
  2. `b12x`'s low-latency `MoETinyDecodeKernelBackend` only supported `"silu"`, falling back to `MoEDynamicKernelSitu` even at `M = 1..3`.
- **Fix**:
  1. Forwards `*args, **kwargs` in `MoEDynamicKernelSitu.__init__`.
  2. Implements exact `SiTU` (`beta = 4.0`, `lbeta = 25.0`, `fastmath=False`) in `tiny_decode.py` and enables `TinyDecode-SITU` for `1 <= M <= 3` (`+1.08 ms/step` faster on GB10) while retaining `MoEDynamicKernelSitu` for `M >= 4` (where `Dynamic`'s shared-expert deduplication wins on 8-token speculative verification batches).

### Patch 8 — Zero-Copy In-Kernel `OUT_HEADS = 16` `FlashIndexer` Padding
- **Problem**: Consumer/workstation Blackwell `DeepGEMM` `fp8_mqa_logits` asserts `num_heads in {16, 32, 64}`, whereas `FlashIndexer` gathers `TOPK = 8` heads per token. Naive `torch.cat` zero-padding allocates and copies 3 tensors (`q_sel`, `w_sel`, `q_sf_sel`) on every prefill chunk.
- **Fix**: Pre-allocates zeroed `OUT_HEADS = 16` output buffers on `cap.major == 12` and writes the `8` gathered heads directly at stride `OUT_HEADS` inside `_flashindexer_fused_gemm_kernel` (`0.0 * ReLU(0) == 0.0`, mathematically identical MQA logits, zero extra kernel launches, preserving `launch_pdl=True` overlap).

### Patch 9 — Full-Vocabulary (`201,216`) Fused FP8 `DSparkMarkovHead` + In-Register `Multimax`
- **Problem**: Across `k = 7` speculative draft steps, `DSpark` ran 7× dense BF16 `[B, 512] @ [512, 201216]` GEMMs (`206 MB` streamed per draft step = `1.44 GB` of LPDDR5X bandwidth per target step!) plus 7× separate `isneginf()` + `_multimax_kernel` + `masked_fill_()` passes over `201,216` logits (`~8.2 ms/step`). Meanwhile, candidate-truncating `--dspark-draft-topk 512` hurt sampled acceptance because `markov_w2` transitions can promote tokens outside the base draft head's top-512.
- **Fix**:
  - Quantizes `markov_w2` post-load to per-row `float8_e4m3fn` (`_markov_w2_fp8: [201216, 512]`, `103 MB` vs `206 MB`, cosine similarity `0.99974`).
  - Fuses `markov_w1[prev_ids]` gather + `markov_w2` FP8 dot product + `base_logits` addition + piecewise-quadratic `Multimax` + `-inf` preservation into a single Triton kernel (`_fused_fp8_markov_multimax_kernel`) writing into a persistent CUDA-graph-safe buffer (`_fused_logits_buf`).
  - Reduces 7-step Markov+Multimax latency from `~8.2 ms` to **`1.6–2.6 ms`** across the **entire 201,216 vocabulary** (`--dspark-draft-topk 0`) while freeing enough unified RAM to raise `--gpu-memory-utilization` from `0.84` (`13.85 GiB` KV) to **`0.86` (`17.14 GiB` KV, `+23.8%`)** — unlocking **`501.6 ms` (`32.9×` faster)** warm 28K prefix-cache hits!

---

## 3. RoCEv2 Collective Tuning (`NCCL_PROTO=LL,Simple`)

Direct NCCL microbenchmarks across Dual DGX Spark's ConnectX-7 200GbE RoCEv2 link revealed a sharp PCIe root-complex latency cliff in `LL128` at `16–64 KB` message sizes (the exact payload window of `TP = 2` decode/verification collectives). Setting:

```bash
NCCL_PROTO=LL,Simple
NCCL_MAX_NCHANNELS=4
NCCL_MIN_NCHANNELS=1
NCCL_IB_QPS_PER_CONNECTION=1
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0
```

eliminates `LL128` stalls and minimizes per-collective launch overhead across all `92+` `AllReduce` operations per step.

---

## 4. Documented Negative Results (What Not to Enable)

1. **Hybrid Shape-Dependent `VHAPostMix` Dtype (`fp32` for `M<=128` / `bf16` for `M>128`)**:
   - Diverging tensor dtypes between `profile_run` (`8,192` tokens, `bf16`) and piecewise CUDA graph capture (`<=128` tokens, `fp32`) triggers `cudaErrorIllegalAddress` during graph replay alongside `GatedDeepGemmLinear`. Keep `VHAPostMix` uniformly `bfloat16` across all batch sizes.
2. **Duplicating Cloned BF16 Markov Weights (`+206 MB`)**:
   - Allocating extra BF16 copies of `markov_w1`/`markov_w2` shrinks per-node KV cache from `17.14 GiB` to `16.38 GiB` (evicting 28K prefix caches under multi-turn load) and is `~2 ms/step` slower than per-row `FP8` on LPDDR5X bandwidth, with `0.00` gain in `DSpark` acceptance (`5.16 tok/step` in both).
3. **`--adaptive-verification` (`enable_confidence_head=True`)**:
   - Adds confidence-head overhead and bypasses the fused 7-step Markov+Multimax fast path without improving wall-clock tok/s at `C = 1..8`. Leave `ADAPTIVE_VERIFICATION=0` (default).
