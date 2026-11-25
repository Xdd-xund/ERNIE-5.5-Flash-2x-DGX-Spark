# Troubleshooting Guide (2× NVIDIA DGX Spark)

Common hardware, networking, memory, and JIT compilation issues when deploying `baidu/ERNIE-5.5-Flash-Preview` across two NVIDIA DGX Spark nodes.

---

## 1. ConnectX-7 Port 0 vs. Port 1 Interface & RDMA Naming

Each NVIDIA DGX Spark exposes **two** physical QSFP/OSFP 200GbE ports on its ConnectX-7 adapter, and each physical port appears as **two** PCIe root-complex RDMA devices (`rocep1s0f*` and `roceP2p1s0f*`):

| Physical Cable Port | `FABRIC_IFNAME` | `RDMA_HCAS` |
| :--- | :--- | :--- |
| **Port 0** (default in `.env.example`) | `enp1s0f0np0` | `rocep1s0f0,roceP2p1s0f0` |
| **Port 1** | `enp1s0f1np1` | `rocep1s0f1,roceP2p1s0f1` |

Verify which port has link and IP on both nodes:
```bash
rdma link
ip -4 -br addr
```
Then update `FABRIC_IFNAME` and `RDMA_HCAS` in `.env` and re-run `./scripts/check_cluster_prereqs.sh`.

---

## 2. NCCL RoCEv2 Hang or `NCCL WARN Call to connect returned Connection refused`

1. **Verify `NCCL_IB_GID_INDEX` (`3` on Ubuntu 24.04 / DGX OS)**:
   ```bash
   grep -H . /sys/class/infiniband/*/ports/1/gid_attrs/types/* 2>/dev/null
   cat /sys/class/infiniband/*/ports/1/gids/3
   ```
   The selected GID index must have type **`RoCE v2`** and contain your IPv4 fabric address (`::ffff:10.42.0.x`).
2. **Verify Subnet Prefix Length (`FABRIC_PREFIX_LEN`)**:
   - If your direct-attached cable uses `/24` (`10.42.0.1/24` and `10.42.0.2/24`), keep `FABRIC_PREFIX_LEN=24`.
   - If you configured `/30` or `/31` point-to-point addressing, set `FABRIC_PREFIX_LEN=30` (or `31`) in `.env` so `NCCL_IB_SUBNET_PREFIX_LEN` matches your subnet mask.
3. **Firewall (`ufw`)**:
   - Ensure traffic on `FABRIC_IFNAME` (especially TCP port `50055` for PyTorch distributed store + ephemeral Gloo/NCCL bootstrap ports) is unrestricted between Node 0 and Node 1:
     ```bash
     sudo ufw allow in on enp1s0f0np0
     ```

---

## 3. First Boot Takes 8–12 Minutes ("Is It Stuck?")

On the **very first launch** with an empty `CACHE_DIR` (`/var/tmp/vllm-ernie55-sm121`), `SM121` (`GB10`) JIT-compiles:
- `FlashInfer` `SM120`/`SM121` sparse MLA & HCA decode/prefill kernels
- `DeepGEMM` `sm121_*` FP8 / UE8M0 / MQA logits kernels (`nvcc`)
- `b12x` CuTe DSL `TinyDecode-SITU` and `MoEDynamicKernelSitu` kernels
- `Triton` `_fused_fp8_markov_multimax_kernel` & `_flashindexer_fused_gemm_kernel`

Monitor live compilation progress with:
```bash
python3 scripts/launch_cluster.py logs -f
```
Because `CACHE_DIR` and `NODE1_CACHE_DIR` are volume-mounted from the host (`-v ${CACHE_DIR}:/root/.cache/vllm`), **every subsequent restart skips JIT compilation and reaches `READY` in ~75–95 seconds**.

---

## 4. Unified Memory OOM During Startup (`128 GiB` LPDDR5X)

On DGX Spark, CPU RAM and GPU VRAM share the same `128 GiB` LPDDR5X pool (`~121.7 GiB` usable by Linux).
`ERNIE-5.5-Flash-Preview` (`TP=2`) requires `~77.3 GiB` of weights + `~17.1 GiB` of FP8 KV cache + `~19 GiB` of runtime/CUDA-graph workspaces per node (`~114–115 GiB` total).

If startup fails with `cudaErrorMemoryAllocation` or Linux OOM-killer:
1. Check competing processes on **both** nodes (`nvidia-smi` and `free -h`). Stop any other LLM containers, Jupyter kernels, or desktop browser sessions consuming memory.
2. Drop stale clean OS page cache before starting:
   ```bash
   python3 scripts/launch_cluster.py start --drop-page-cache --wait
   ```
3. If your system runs heavy background host services (~5–8 GiB resident), lower `GPU_MEMORY_UTILIZATION` in `.env` from `0.86` to `0.84` (or `0.82`).

---

## 5. `FlashInfer` Cubin Architecture Mismatch (`arch=121` / `sm_90a`)

If you customize the `Dockerfile` and install extra packages that pull `flashinfer-cubin` or `flashinfer-jit-cache` back into `/usr/local/lib/python3.12/dist-packages`, `FlashInfer` may attempt to load `SM90`/`SM100` pre-compiled cubins instead of JIT-compiling for `SM121`.
Ensure both packages remain uninstalled inside the image:
```bash
uv pip uninstall --system flashinfer-cubin flashinfer-jit-cache
```
(Already handled automatically in [`Dockerfile`](../Dockerfile).)
