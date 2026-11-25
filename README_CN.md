# 双机 NVIDIA DGX Spark (`GB10 SM121`) 部署与加速 `ERNIE-5.5-Flash-Preview` 完整指南

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-ERNIE--5.5--Flash--Preview_(280B_A14B)-green)](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)
[![Hardware](https://img.shields.io/badge/Hardware-2%C3%97_NVIDIA_DGX_Spark_(GB10_SM121)-76b900)](docs/ARCHITECTURE_AND_PATCHES.md)

**[English](README.md) | [简体中文](README_CN.md) | [架构与算子补丁详解](docs/ARCHITECTURE_AND_PATCHES.md) | [常见问题排查](docs/TROUBLESHOOTING.md) | [版本记录](CHANGELOG.md)**

本仓库提供一套**开箱即用、100% 可复刻**的容器构建、GB10 (`SM121`) 底层算子适配与提速补丁、200GbE ConnectX-7 RoCEv2 双机 `TP=2` 集群编排及基准评测方案，用于在 **2 台 NVIDIA DGX Spark (`Grace-Blackwell GB10`，单机 `128 GiB` LPDDR5X 统一内存)** 上满血运行百度开源旗舰模型 **[`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)**（总参数 `280B` / 激活 `14B`，W4A8 MXFP4 MoE + FP8 HCA/MLA-DSA 混合稀疏注意力 + `FlashIndexer` + 内置 3 层 `DSpark` 投机解码，支持 **`131,072` (128K)** 上下文）。

---

## 为什么需要本仓库？（原生镜像在 DGX Spark 上的 6 大阻断项）

官方上游 `vLLM 0.32.0` + `baidu/vLLM-ERNIE` 默认面向数据中心 `SM90`/`SM100`（H800 / B200 `FlashMLA`）设计，在 DGX Spark（`DeviceCapability(12, 1)` / `SM121` + `128 GiB` CPU/GPU 共享统一内存）上直接启动会遇到 6 处硬性阻断或严重瓶颈：

1. **`FlashMLA` 不支持 `SM121`**：开启 `--kv-cache-dtype fp8` 时，上游将 `40` 层 HCA 与 `6` 层 MLA-DSA 硬编码路由至仅支持 `SM90`/`SM100` 的 `FlashMLA`。
2. **`DeepGEMM` 缺少 `sm121_*.cuh` 头文件**：`SM121` JIT 编译时查找 `sm121_*.cuh`，而上游仅包含 `sm120_*.cuh`。
3. **`FlashIndexer` 8 头在 Blackwell 上触发断言**：`SM120`/`SM121` 的 `fp8_mqa_logits` 强制要求 `num_heads ∈ {16, 32, 64}`，而 `FlashIndexer` 每 token 选取 `8` 个头。
4. **`b12x` W4A8 MXFP4 `SiTU` 构造器报错且未启用 `TinyDecode`**：`MoEDynamicKernelSitu.__init__` 漏传 `*args, **kwargs`，且 `TinyDecode` 仅放行 `silu`。
5. **128 GiB 统一内存加载权重时 Page Cache 挤爆 OOM**：每节点读取 `77.3 GiB` safetensors 分片时，Linux 文件页缓存直接占用同一块 `128 GiB` LPDDR5X 物理内存池。
6. **跨机 RoCEv2 与 `DSpark` 投机头带宽瓶颈**：`VHAPostMix` 默认升 `fp32` 跨机 `AllReduce`、`GatedDeepGemmLinear` 锁死 `TP=1`、PCIe `LL128` 协议延迟尖峰，以及 7 步 `DSparkMarkovHead` 每步搬运 `1.44 GB` BF16 词表矩阵并挤压 KV Cache 容量。

本仓库通过构建期幂等补丁 [`patches/patch_gb10_ernie55.py`](patches/patch_gb10_ernie55.py) 与启动器 [`scripts/launch_cluster.py`](scripts/launch_cluster.py) **一次性无损解决全部问题**（通过 4 项严格数学/代码/JSON/并发工具调用门禁验证）。

---

## 实测性能对比（双机 DGX Spark `TP=2`，`131K` 上下文）

测试脚本见 [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py)，原始评测报告保存在 [`benchmarks/results/`](benchmarks/results/)：

| 评测指标 / 负载场景 | `v0.1.0`（首次跑通基线） | `v0.3.0`（深度调优版） | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **单节点可用 FP8 KV Cache 容量** | `13.85 GiB` (`util=0.84`) | **`17.14 GiB`** (`util=0.86`) | **`+23.8%`**（双机各留 `~6.5 GiB` 安全余量） |
| **Warm `28K` 前缀缓存命中首字延迟 (TTFT)** | `16,506.2 ms` (`1,690.9 tok/s`，被驱逐) | **`501.6 ms` (`55,625.5 tok/s`)** | **首字提速 `32.9×`** |
| **Cold `7K` 长文本预填充 (Prefill)** | `1,714.9 tok/s` (`4.08 s`) | **`2,013.6 tok/s` (`3.47 s`)** | **`+17.4%`** |
| **Cold `28K` 长文本预填充 (Prefill)** | `1,680.9 tok/s` (`16.60 s`) | **`1,976.0 tok/s` (`14.12 s`)** | **`+17.6%`** |
| **单路解码速度 (`C=1`, Greedy `T=0.0`)** | `60.33 tok/s` (`5.16 tok/step`) | **`66.75 tok/s` (`5.54 tok/step`)** | **`+10.6%`** |
| **单路数学推理 (`7^2026+11^2026 mod 100`)** | `59.17 tok/s` | **`61.30 tok/s`** | **`+3.6%`**（精确输出 `10`） |
| **单路采样对话 (`C=1`, Sampled `T=0.6`)** | `27.88 tok/s` | **`29.32 tok/s`** | **`+5.2%`** |
| **双并发总吞吐 (`C=2`, Sampled `T=0.6`)** | `84.59 tok/s` | **`90.51 tok/s`** | **`+7.0%`** |
| **四并发总吞吐 (`C=4`, Sampled `T=0.6`)** | `69.67 tok/s` | **`74.86 tok/s`** | **`+7.4%`** |
| **八并发峰值吞吐 (`C=8`, Sampled `T=0.6`)** | `103.88 tok/s` | **`118.07 tok/s`** | **`+13.7%`** |
| **八并发贪婪吞吐 (`C=8`, Greedy `T=0.0`)** | `110.39 tok/s` | **`115.16 tok/s`** | **`+4.3%`** |
| **无损质量门禁 (`Quality Gate`)** | `4 / 4` 全通过 | **`4 / 4` 全通过** | **零精度损失** |

<sub>*数据来源说明：`v0.1.0` 对应 `benchmarks/results/r0_v010_baseline.json`；`v0.3.0` 对应连续两轮全矩阵实测 `r2_v030_run1.json`（单路及 C=1/2/8 并发）与 `r2_v030_warm_prefix.json`（C=4 采样及冷/热长上下文预填充）。*</sub>

---

## 硬件与环境要求

1. **硬件**：2 台 NVIDIA DGX Spark（`GB10` ARM64，每台 `128 GiB` 统一内存），通过 **ConnectX-7 200GbE** 高速线缆直连或交换机互联。
2. **系统环境**：DGX OS / Ubuntu 24.04 (`aarch64`)，已安装 NVIDIA CUDA 13.0+ 驱动、Docker、`nvidia-container-toolkit` 以及 `rdma-core`。
3. **存储空间**：每台节点需预留约 `160 GiB` 存放模型权重 + 约 `35 GiB` 存放容器镜像与 SM121 JIT 编译缓存。
4. **免密 SSH**：主节点（Node 0）已配置到从节点（Node 1）的免密 SSH 登录（`ssh-copy-id`）。

---

## 4 步完整复刻流程（在主节点 Node 0 上执行）

### 第 1 步：克隆仓库并配置 `.env`

```bash
git clone https://github.com/Xdd-xund/ERNIE-5.5-Flash-2x-DGX-Spark.git
cd ERNIE-5.5-Flash-2x-DGX-Spark
cp .env.example .env
```

按你的双机实际环境修改 `.env`（重点确认双机高速网口 IP、SSH 用户名及 ConnectX-7 插线端口）：

```ini
NODE0_IP=10.42.0.1
NODE1_IP=10.42.0.2
NODE1_SSH=ubuntu@10.42.0.2

# 线缆插在 ConnectX-7 Port 0 -> FABRIC_IFNAME=enp1s0f0np0 | RDMA_HCAS=rocep1s0f0,roceP2p1s0f0
# 线缆插在 ConnectX-7 Port 1 -> FABRIC_IFNAME=enp1s0f1np1 | RDMA_HCAS=rocep1s0f1,roceP2p1s0f1
FABRIC_IFNAME=enp1s0f0np0
RDMA_HCAS=rocep1s0f0,roceP2p1s0f0
NCCL_IB_GID_INDEX=3

MODEL_DIR=/data/models/ERNIE-5.5-Flash-Preview
NODE1_MODEL_DIR=/data/models/ERNIE-5.5-Flash-Preview
```

运行预检脚本确认双机 GPU、内存、RoCEv2 GID 及 SSH 全部就绪：

```bash
./scripts/check_cluster_prereqs.sh --skip-weights
```

### 第 2 步：下载并同步模型权重（`~154.5 GiB`，`128` 个分片 + `flashindexer/`）

```bash
# 国内环境推荐使用 ModelScope 下载并自动通过 200GbE 高速网同步至 Node 1：
./scripts/prepare_weights.sh --modelscope

# 或使用 HuggingFace 官方 CLI 下载（支持在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com）：
./scripts/prepare_weights.sh
```

> 若主节点本地已有权重，只需在 `.env` 中将 `MODEL_DIR` 指向该目录；若 Node 1 尚未拷贝，执行 `./scripts/prepare_weights.sh --sync-only` 即可利用 200GbE 链路极速同步，随后运行 `./scripts/check_cluster_prereqs.sh` 复核。

### 第 3 步：构建 `v0.3.0` 容器镜像并同步到 Node 1

基于官方公开镜像 `vllm/vllm-openai:v0.32.0` 一键构建，自动打入并校验全部 9 项 `SM121` 补丁，随后通过 200GbE 链路流式导入 Node 1：

```bash
./scripts/build_image.sh
```

### 第 4 步：一键拉起双机 `TP=2` 服务

```bash
# （可选）先打印双机完整 docker run 命令进行审阅：
python3 scripts/launch_cluster.py print-cmd

# 启动双机容器并自动等待 /health 就绪：
python3 scripts/launch_cluster.py start --wait
```

> **首次启动提示**：首次在空缓存目录（`CACHE_DIR`）启动时，系统会为 `GB10 (SM121)` 现场 JIT 编译 `FlashInfer`、`DeepGEMM`、`b12x` 和 `Triton` 算子，约需 **8～12 分钟**；编译产物会自动持久化在宿主机 `CACHE_DIR` 中，**此后每次重启仅需 75～95 秒即可就绪**。

常用运维命令：
```bash
python3 scripts/launch_cluster.py status          # 查看双机容器与健康状态
python3 scripts/launch_cluster.py logs -f         # 实时查看主节点日志（加 --rank 1 查看从节点）
python3 scripts/launch_cluster.py stop            # 停止并清理双机容器
```

---

## OpenAI 兼容接口调用与复现评测

服务默认监听 `http://<NODE0_IP>:8898/v1`，完整支持思考模式（`enable_thinking` + `ernie5` 推理解析器）与原生并发工具调用（`ernie5` tool parser）：

```bash
curl http://127.0.0.1:8898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "baidu/ERNIE-5.5-Flash-Preview",
    "messages": [
      {"role": "user", "content": "求 7^2026 + 11^2026 除以 100 的余数。"}
    ],
    "temperature": 0.6,
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": true},
    "reasoning_effort": "low"
  }'
```

运行完整吞吐基准与 4 项无损质量门禁：

```bash
python3 benchmarks/run_benchmark.py \
  --base-url http://127.0.0.1:8898 \
  --label reproduce_v030 \
  --out benchmarks/results/local_reproduce_v030.json \
  --verify-quality
```

---

## 开源许可与致谢

- 本仓库脚本与适配补丁采用 **[Apache License 2.0](LICENSE)** 开源协议发布。
- 上游依赖与第三方项目声明详见 [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)。
