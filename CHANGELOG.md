# Changelog

All notable engineering milestones, kernel adaptations, and verified benchmark progressions for running `baidu/ERNIE-5.5-Flash-Preview` (`280B` total / `14B` active, W4A8 MXFP4 MoE + FP8 KV + `DSpark` + `FlashIndexer`) on **2× NVIDIA DGX Spark (`GB10 SM121`)**.

---

## [v0.3.0] — Production Speed & KV-Cache Optimization Release

### Highlights (vs. `v0.1.0` Initial Adaptation Baseline)
- **Warm 28K Prefix-Cache TTFT**: `16,506.2 ms` (`1,690.9 tok/s`, evicted) $\rightarrow$ **`501.6 ms` (`55,625.5 tok/s`, `32.9×` faster)**
- **KV Cache Capacity per Node**: `13.85 GiB` $\rightarrow$ **`17.14 GiB` (`+23.8%`)**, with `6.54–6.88 GiB` free OS memory headroom on both `128 GiB` nodes (`0` OOM)
- **Cold Long-Context Prefill**:
  - `7K` prompt: `1,714.9` $\rightarrow$ **`2,013.6 tok/s` (`+17.4%`)**
  - `28K` prompt: `1,680.9` $\rightarrow$ **`1,976.0 tok/s` (`+17.6%`, `16.60 s` $\rightarrow$ `14.12 s`)**
- **Single-Stream & Multi-Stream Decode**:
  - `Greedy C=1` decode: `60.33` $\rightarrow$ **`66.75 tok/s` (`+10.6%`)**
  - `Sampled (t=0.6) C=2`: `84.59` $\rightarrow$ **`90.51 tok/s` (`+7.0%`)**
  - `Sampled (t=0.6) C=4`: `69.67` $\rightarrow$ **`74.86 tok/s` (`+7.4%`)**
  - `Sampled (t=0.6) C=8` peak throughput: `103.88` $\rightarrow$ **`118.07 tok/s` (`+13.7%`)**
  - `Greedy C=8` aggregate throughput: `110.39` $\rightarrow$ **`115.16 tok/s` (`+4.3%`)**
- **Lossless Quality Gate**: `4/4` passed (`7^2026 + 11^2026 mod 100 == 10`, `LRUCacheWithTTL` unit tests, strict raw JSON, parallel tool calling).

### Added & Optimized
- **Full-Vocabulary (`201,216`) Fused FP8 `DSparkMarkovHead` + In-Register `Multimax` Kernel** (`nvidia/dspark.py`, `dspark/speculator.py`, `patches.py`):
  - Quantizes `markov_w2` (`[201216, 512]`) to per-row `float8_e4m3fn` (`103 MB` vs `206 MB` BF16, cosine similarity `0.99974`) and fuses 7-step Markov embedding lookup, `w2` dot product, piecewise-quadratic `Multimax`, and `-inf` preservation into a single Triton kernel (`_fused_fp8_markov_multimax_kernel`), cutting 7-step draft overhead from `8.2 ms` to `1.6–2.6 ms` with **zero candidate truncation** (`--dspark-draft-topk 0`).
- **Native `bfloat16` `VHAPostMix` AllReduce + Contiguous `Ut` Pre-Folding** (`nvidia/attention.py`):
  - Pre-folds contiguous `U^T` buffers at weight load time and executes cross-node VHA `AllReduce` in native `bfloat16` (matching `TP=1` precision, `cos = 0.9999994` vs `fp32`), halving `68%` of per-step cross-node RoCEv2 collective payload across all 46 layers (`40` HCA + `6` MLA-DSA).
- **`TP=2` Unlock for `GatedDeepGemmLinear`** (`nvidia/attention.py`):
  - Removes the `tp_size == 1` guard on `GatedDeepGemmLinear` (`_sigmoid_mul_quant_kernel`), fusing `sigmoid + mul + UE8M0 FP8 block quant` with `RowParallelLinear` `AllReduce` across all 46 attention `wo_b` projections (`diff = 0.0`, bit-exact).
- **`b12x` `TinyDecode-SITU` for `1 <= M <= 3`** (`b12x/moe/_shared/kernels/tiny_decode.py`, `fused_moe/_impl.py`):
  - Adds exact `SiTU` (`beta=4.0, lbeta=25.0, fastmath=False`) epilogue support to `MoETinyDecodeKernelBackend` for `1 <= M <= 3` (`+1.08 ms/step` faster on GB10) while preserving `MoEDynamicKernelSitu` expert deduplication for `M >= 4` (speculative verification batches of `8` tokens).
- **Zero-Copy In-Kernel `OUT_HEADS=16` `FlashIndexer` Padding** (`nvidia/flashindexer.py`):
  - Eliminates 3× per-chunk `torch.cat` tensor copies by writing `8` selected heads directly into pre-zeroed `16`-head stride buffers inside `_flashindexer_fused_gemm_kernel` while preserving Blackwell `launch_pdl=True` overlap.
- **RoCEv2 Low-Latency NCCL Profile** (`scripts/launch_cluster.py`):
  - Sets `NCCL_PROTO=LL,Simple`, `NCCL_MAX_NCHANNELS=4`, `NCCL_MIN_NCHANNELS=1`, `NCCL_IB_QPS_PER_CONNECTION=1` (avoiding PCIe root-complex `LL128` regressions at `16–64 KB` message sizes on GB10).

---

## [v0.2.0] — Microbenchmark & Draft Top-K Exploration (Superseded by `v0.3.0`)
- Validated RoCEv2 collective microbenchmarks (`NCCL_PROTO=LL,Simple`) and `b12x` `TinyDecode` vs `Dynamic` crossover (`M <= 3` vs `M >= 4`).
- Observed that candidate-truncated `--dspark-draft-topk 512` reduced speculative acceptance on sampled prompts (`t=0.6`) because `markov_w2` transitions can promote tokens outside the base draft model's top-512 — motivating `v0.3.0`'s full-vocabulary (`201,216`) fused FP8 kernel.

---

## [v0.1.0] — Initial GB10 (`SM121`) Hardware Adaptation Baseline
- Enabled first end-to-end `TP=2` boot and serving of `baidu/ERNIE-5.5-Flash-Preview` across 2× NVIDIA DGX Spark over 200GbE ConnectX-7 RoCEv2:
  1. `DeepGEMM` `sm121_*.cuh` $\rightarrow$ `sm120_*.cuh` kernel header aliases (`11` kernels).
  2. `Ernie5_5HCAFlashInferSM120Attention` & `Ernie5_5FlashInferMLASparseSM120Backend` with per-head `attn_sink` forwarding for `DeviceCapability(12, 1)`.
  3. `b12x` `MoEDynamicKernelSitu(*args, **kwargs)` constructor fix.
  4. `FlashIndexer` `8 -> 16` head zero-padding for SM120/SM121 `fp8_mqa_logits`.
  5. `POSIX_FADV_DONTNEED` per-shard safetensors page-cache drop (`MBX_LOAD_DROP_CACHE=1`) for `128 GiB` LPDDR5X unified memory.
  6. Grace ARM64 P-core `shm_broadcast.py` spin-wait reduction (`busy_loop_s: 1 -> 0.002`).
