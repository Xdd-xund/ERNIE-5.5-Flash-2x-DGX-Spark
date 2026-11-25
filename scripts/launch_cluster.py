#!/usr/bin/env python3
"""
Turnkey Dual NVIDIA DGX Spark (`2x GB10 SM121`, `2x 128 GiB`, Dual 200GbE RoCEv2)
Cluster Controller for `baidu/ERNIE-5.5-Flash-Preview` (`ernie55-dual-spark:v0.3.0`).

Supported actions:
  - `print-cmd` : Print exact Rank 0 & Rank 1 `docker run` commands (dry-run).
  - `start`     : Launch Rank 1 (headless worker via SSH) + Rank 0 (OpenAI server),
                  with optional `--wait` readiness polling and `--drop-page-cache`.
  - `stop`      : Stop and remove containers on both nodes.
  - `status`    : Show container status and `/health` API readiness state.
  - `logs`      : Show or follow container logs (`--rank 0` or `--rank 1`, `-f`).
  - `health`    : Query `/health` and `/v1/models` on Rank 0.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _strip_inline_comment(val: str) -> str:
    in_quote: str | None = None
    for idx, ch in enumerate(val):
        if ch in ("'", '"') and (idx == 0 or val[idx - 1] != "\\"):
            in_quote = None if in_quote == ch else (in_quote or ch)
        elif ch == "#" and in_quote is None and (idx == 0 or val[idx - 1].isspace()):
            return val[:idx].strip()
    return val.strip()


def load_dotenv(env_path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = _strip_inline_comment(val.strip())
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        loaded[key] = val
    return loaded


def cfg_str(env_map: dict[str, str], key: str, default: str) -> str:
    env_val = os.environ.get(key)
    if env_val is not None and env_val != "":
        return env_val
    file_val = env_map.get(key)
    if file_val is not None and file_val != "":
        return file_val
    return default


def cfg_int(env_map: dict[str, str], key: str, default: int) -> int:
    return int(cfg_str(env_map, key, str(default)))


def cfg_float(env_map: dict[str, str], key: str, default: float) -> float:
    return float(cfg_str(env_map, key, str(default)))


def cfg_bool(env_map: dict[str, str], key: str, default: bool) -> bool:
    raw = cfg_str(env_map, key, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def run_local(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def run_node1(node1_ssh: str, remote_cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node1_ssh, remote_cmd],
        text=True,
        capture_output=True,
        check=check,
    )


def require_ok(res: subprocess.CompletedProcess, label: str) -> subprocess.CompletedProcess:
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "").strip()
        sys.exit(f"ERROR: {label} failed (exit {res.returncode}):\n{detail}")
    return res


def build_docker_cmd(
    *,
    rank: int,
    host_ip: str,
    master_ip: str,
    master_port: int,
    fabric_ifname: str,
    fabric_prefix_len: int,
    rdma_hcas: str,
    ib_gid_index: int,
    container_name: str,
    model_dir: str,
    cache_dir: str,
    image: str,
    served_model_name: str,
    bind_host: str,
    port: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    gpu_mem_util: float,
    load_format: str,
    kv_cache_dtype: str,
    moe_backend: str,
    enable_dspark: bool,
    num_spec_tokens: int,
    dspark_draft_topk: int,
    adaptive_verification: bool,
    enable_flashindexer: bool,
) -> list[str]:
    env_vars = {
        "VLLM_HOST_IP": host_ip,
        "GLOO_SOCKET_IFNAME": fabric_ifname,
        "TP_SOCKET_IFNAME": fabric_ifname,
        "NCCL_SOCKET_IFNAME": fabric_ifname,
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": rdma_hcas,
        "NCCL_IB_GID_INDEX": str(ib_gid_index),
        "NCCL_IB_ROCE_VERSION_NUM": "2",
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
        "NCCL_IB_SUBNET_PREFIX_LEN": str(fabric_prefix_len),
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_CROSS_NIC": "0",
        "NCCL_IB_AUTO_DETECT": "0",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_PROTO": "LL,Simple",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_IB_QPS_PER_CONNECTION": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "OMP_NUM_THREADS": "8",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "300",
        "MBX_LOAD_DROP_CACHE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "XDG_CACHE_HOME": "/root/.cache/vllm/xdg",
        "FLASHINFER_WORKSPACE_BASE": "/root/.cache/vllm/flashinfer",
        "DG_JIT_CACHE_DIR": "/root/.cache/vllm/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/vllm/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/vllm/inductor",
        "VLLM_CACHE_ROOT": "/root/.cache/vllm",
        "PYTHONUNBUFFERED": "1",
    }

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--device",
        "/dev/infiniband:/dev/infiniband:rwm",
        "--cap-add",
        "CAP_IPC_LOCK",
        "--cap-add",
        "CAP_SYS_NICE",
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "stack=67108864:67108864",
        "-v",
        f"{model_dir}:/opt/ernie/model:ro",
        "-v",
        f"{cache_dir}:/root/.cache/vllm",
    ]
    for k, v in env_vars.items():
        cmd += ["-e", f"{k}={v}"]

    vllm_args = [
        image,
        "/opt/ernie/model",
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--tensor-parallel-size",
        "2",
        "--nnodes",
        "2",
        "--node-rank",
        str(rank),
        "--master-addr",
        master_ip,
        "--master-port",
        str(master_port),
        "--distributed-executor-backend",
        "mp",
        "--moe-backend",
        moe_backend,
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--load-format",
        load_format,
        "--safetensors-load-strategy",
        "lazy",
        "--enable-prefix-caching",
        "--reasoning-parser",
        "ernie5",
        "--tool-call-parser",
        "ernie5",
        "--enable-auto-tool-choice",
        "--host",
        bind_host,
        "--port",
        str(port),
    ]

    hf_overrides: dict[str, object] = {}
    if enable_flashindexer:
        hf_overrides["enable_flashindexer"] = True
    if enable_dspark and adaptive_verification:
        hf_overrides["enable_confidence_head"] = True
    if hf_overrides:
        vllm_args += ["--hf-overrides", json.dumps(hf_overrides)]

    if enable_dspark:
        spec_cfg: dict[str, object] = {
            "method": "dspark",
            "num_speculative_tokens": num_spec_tokens,
            "draft_sample_method": "probabilistic",
        }
        if dspark_draft_topk > 0:
            spec_cfg["dspark_draft_topk"] = dspark_draft_topk
        if adaptive_verification:
            spec_cfg["enable_adaptive_verification"] = True
        vllm_args += ["--speculative-config", json.dumps(spec_cfg)]

    if rank > 0:
        vllm_args.append("--headless")
    return cmd + vllm_args


def check_http_ready(port: int, bind_host: str = "0.0.0.0", timeout_s: float = 3.0) -> tuple[bool, str]:
    probe_host = "127.0.0.1" if bind_host in ("", "0.0.0.0", "::") else bind_host
    if ":" in probe_host and not probe_host.startswith("["):
        probe_host = f"[{probe_host}]"
    base = f"http://{probe_host}:{port}"
    try:
        with _DIRECT_OPENER.open(f"{base}/health", timeout=timeout_s) as r:
            if r.status != 200:
                return False, f"HTTP {r.status}"
        with _DIRECT_OPENER.open(f"{base}/v1/models", timeout=timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
            models = [m.get("id", "?") for m in data.get("data", [])]
            return True, f"READY at {base} (models={models})"
    except Exception as exc:
        return False, str(exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    known, _ = pre.parse_known_args(argv)
    env_map = load_dotenv(Path(known.env_file))

    ap = argparse.ArgumentParser(
        description="Turnkey Dual DGX Spark TP=2 Launcher for baidu/ERNIE-5.5-Flash-Preview"
    )
    ap.add_argument(
        "action",
        choices=["start", "stop", "status", "logs", "health", "print-cmd"],
        help="Cluster lifecycle action",
    )
    ap.add_argument("--env-file", default=known.env_file, help="Path to .env configuration file")
    ap.add_argument("--image", default=cfg_str(env_map, "IMAGE", "ernie55-dual-spark:v0.3.0"))
    ap.add_argument("--container-name", default=cfg_str(env_map, "CONTAINER_NAME", "ernie55-dual-spark"))
    ap.add_argument("--node0-ip", default=cfg_str(env_map, "NODE0_IP", "10.42.0.1"))
    ap.add_argument("--node1-ip", default=cfg_str(env_map, "NODE1_IP", "10.42.0.2"))
    ap.add_argument("--node1-ssh", default=cfg_str(env_map, "NODE1_SSH", "ubuntu@10.42.0.2"))

    ifname_default = cfg_str(env_map, "FABRIC_IFNAME", "enp1s0f0np0")
    hcas_default = cfg_str(env_map, "RDMA_HCAS", "rocep1s0f0,roceP2p1s0f0")
    ap.add_argument("--fabric-ifname", default=ifname_default)
    ap.add_argument("--node1-fabric-ifname", default=cfg_str(env_map, "NODE1_FABRIC_IFNAME", ifname_default))
    ap.add_argument("--fabric-prefix-len", type=int, default=cfg_int(env_map, "FABRIC_PREFIX_LEN", 24))
    ap.add_argument("--rdma-hcas", default=hcas_default)
    ap.add_argument("--node1-rdma-hcas", default=cfg_str(env_map, "NODE1_RDMA_HCAS", hcas_default))
    ap.add_argument("--ib-gid-index", type=int, default=cfg_int(env_map, "NCCL_IB_GID_INDEX", 3))

    model_default = cfg_str(env_map, "MODEL_DIR", "/data/models/ERNIE-5.5-Flash-Preview")
    cache_default = cfg_str(env_map, "CACHE_DIR", "/var/tmp/vllm-ernie55-sm121")
    ap.add_argument("--model-dir", default=model_default)
    ap.add_argument("--node1-model-dir", default=cfg_str(env_map, "NODE1_MODEL_DIR", model_default))
    ap.add_argument("--cache-dir", default=cache_default)
    ap.add_argument("--node1-cache-dir", default=cfg_str(env_map, "NODE1_CACHE_DIR", cache_default))

    ap.add_argument("--served-model-name", default=cfg_str(env_map, "SERVED_MODEL_NAME", "baidu/ERNIE-5.5-Flash-Preview"))
    ap.add_argument("--host", default=cfg_str(env_map, "HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=cfg_int(env_map, "PORT", 8898))
    ap.add_argument("--master-port", type=int, default=cfg_int(env_map, "MASTER_PORT", 50055))
    ap.add_argument("--max-model-len", type=int, default=cfg_int(env_map, "MAX_MODEL_LEN", 131072))
    ap.add_argument("--max-num-seqs", type=int, default=cfg_int(env_map, "MAX_NUM_SEQS", 8))
    ap.add_argument("--max-num-batched-tokens", type=int, default=cfg_int(env_map, "MAX_NUM_BATCHED_TOKENS", 8192))
    ap.add_argument("--gpu-memory-utilization", type=float, default=cfg_float(env_map, "GPU_MEMORY_UTILIZATION", 0.86))
    ap.add_argument(
        "--load-format",
        default=cfg_str(env_map, "LOAD_FORMAT", "auto"),
        choices=["auto", "dummy", "safetensors", "fastsafetensors"],
    )
    ap.add_argument(
        "--kv-cache-dtype",
        default=cfg_str(env_map, "KV_CACHE_DTYPE", "fp8"),
        choices=["fp8", "auto"],
    )
    ap.add_argument(
        "--moe-backend",
        default=cfg_str(env_map, "MOE_BACKEND", "b12x"),
        choices=["b12x", "auto", "marlin", "humming"],
    )

    dspark_default = cfg_bool(env_map, "ENABLE_DSPARK", True)
    fi_default = cfg_bool(env_map, "ENABLE_FLASHINDEXER", True)
    av_default = cfg_bool(env_map, "ADAPTIVE_VERIFICATION", False)

    ap.add_argument("--no-dspark", action="store_true", default=not dspark_default)
    ap.add_argument("--num-spec-tokens", type=int, default=cfg_int(env_map, "NUM_SPEC_TOKENS", 7))
    ap.add_argument("--dspark-draft-topk", type=int, default=cfg_int(env_map, "DSPARK_DRAFT_TOPK", 0))
    ap.add_argument("--adaptive-verification", action="store_true", default=av_default)
    ap.add_argument("--no-adaptive-verification", dest="adaptive_verification", action="store_false")
    ap.add_argument("--no-flashindexer", action="store_true", default=not fi_default)

    ap.add_argument("--wait", action="store_true", help="Wait until /health responds 200 OK after start")
    ap.add_argument("--wait-timeout", type=int, default=1800, help="Max seconds to wait for engine readiness")
    ap.add_argument("--drop-page-cache", action="store_true", help="Drop clean OS page cache via sudo before start")
    ap.add_argument("--rank", type=int, choices=[0, 1], default=0, help="Node rank for `logs` action")
    ap.add_argument("-n", "--tail", type=int, default=80, help="Number of log lines to display")
    ap.add_argument("-f", "--follow", action="store_true", help="Follow container logs")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    common_kw = dict(
        master_ip=args.node0_ip,
        master_port=args.master_port,
        fabric_prefix_len=args.fabric_prefix_len,
        ib_gid_index=args.ib_gid_index,
        container_name=args.container_name,
        image=args.image,
        served_model_name=args.served_model_name,
        bind_host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_mem_util=args.gpu_memory_utilization,
        load_format=args.load_format,
        kv_cache_dtype=args.kv_cache_dtype,
        moe_backend=args.moe_backend,
        enable_dspark=not args.no_dspark,
        num_spec_tokens=args.num_spec_tokens,
        dspark_draft_topk=args.dspark_draft_topk,
        adaptive_verification=args.adaptive_verification,
        enable_flashindexer=not args.no_flashindexer,
    )

    cmd0 = build_docker_cmd(
        rank=0,
        host_ip=args.node0_ip,
        fabric_ifname=args.fabric_ifname,
        rdma_hcas=args.rdma_hcas,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        **common_kw,
    )
    cmd1 = build_docker_cmd(
        rank=1,
        host_ip=args.node1_ip,
        fabric_ifname=args.node1_fabric_ifname,
        rdma_hcas=args.node1_rdma_hcas,
        model_dir=args.node1_model_dir,
        cache_dir=args.node1_cache_dir,
        **common_kw,
    )

    if args.action == "print-cmd":
        print(f"# Rank 0 — Node 0 ({args.node0_ip}, API server on :{args.port}):")
        print(" ".join(shlex.quote(x) for x in cmd0) + "\n")
        print(f"# Rank 1 — Node 1 ({args.node1_ssh} / {args.node1_ip}, headless TP worker):")
        print(" ".join(shlex.quote(x) for x in cmd1))
        return

    if args.action == "stop":
        run_local(["docker", "rm", "-f", args.container_name])
        run_node1(args.node1_ssh, f"docker rm -f {shlex.quote(args.container_name)}")
        print(f"Stopped {args.container_name} on Node 0 ({args.node0_ip}) and Node 1 ({args.node1_ip}).")
        return

    if args.action == "health":
        ok, detail = check_http_ready(args.port, args.host)
        print(f"Health: {detail}")
        sys.exit(0 if ok else 1)

    if args.action == "status":
        flt = f"name=^/{args.container_name}$"
        r0 = run_local(["docker", "ps", "-a", "--filter", flt, "--format", "Node 0: {{.Status}} ({{.Image}})"])
        r1 = run_node1(
            args.node1_ssh,
            f"docker ps -a --filter {shlex.quote(flt)} --format 'Node 1: {{{{.Status}}}} ({{{{.Image}}}})'",
        )
        print(r0.stdout.strip() or "Node 0: not running")
        print(r1.stdout.strip() or "Node 1: not running")
        ok, detail = check_http_ready(args.port, args.host)
        print(f"API   : {detail if ok else 'not ready (' + detail + ')'}")
        return

    if args.action == "logs":
        log_cmd = ["docker", "logs", "--tail", str(args.tail)]
        if args.follow:
            log_cmd.append("-f")
        log_cmd.append(args.container_name)
        if args.rank == 0:
            os.execvp("docker", log_cmd)
        else:
            remote = " ".join(shlex.quote(x) for x in log_cmd)
            os.execvp("ssh", ["ssh", "-o", "BatchMode=yes", args.node1_ssh, remote])
        return

    # --- Action: start ---
    if args.load_format != "dummy":
        if not Path(args.model_dir).is_dir():
            sys.exit(
                f"ERROR: Node 0 model directory not found: {args.model_dir} (run ./scripts/prepare_weights.sh or check .env)"
            )
        chk1 = run_node1(args.node1_ssh, f"test -d {shlex.quote(args.node1_model_dir)}")
        if chk1.returncode != 0:
            sys.exit(f"ERROR: Node 1 model directory not found on {args.node1_ssh}:{args.node1_model_dir}")

    if args.drop_page_cache:
        run_local(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
        run_node1(args.node1_ssh, "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true")

    require_ok(run_local(["mkdir", "-p", args.cache_dir]), f"Node 0 mkdir {args.cache_dir}")
    require_ok(
        run_node1(args.node1_ssh, f"mkdir -p {shlex.quote(args.node1_cache_dir)}"),
        f"Node 1 mkdir {args.node1_cache_dir}",
    )

    run_local(["docker", "rm", "-f", args.container_name])
    run_node1(args.node1_ssh, f"docker rm -f {shlex.quote(args.container_name)}")

    require_ok(
        run_node1(args.node1_ssh, " ".join(shlex.quote(x) for x in cmd1)),
        f"Node 1 ({args.node1_ssh}) docker run",
    )
    res0 = run_local(cmd0)
    if res0.returncode != 0:
        # Clean up orphan Rank 1 container if Rank 0 failed to start
        run_node1(args.node1_ssh, f"docker rm -f {shlex.quote(args.container_name)}")
        require_ok(res0, "Node 0 docker run")

    print(
        f"Started {args.container_name} ({args.image}):\n"
        f"  Rank 1 (worker) : {args.node1_ssh} ({args.node1_ip})\n"
        f"  Rank 0 (server) : http://{args.node0_ip}:{args.port}/v1\n"
        f"  Profile         : TP=2 | kv={args.kv_cache_dtype} | util={args.gpu_memory_utilization} | "
        f"spec_k={args.num_spec_tokens if not args.no_dspark else 0} | draft_topk={args.dspark_draft_topk}"
    )

    if args.wait:
        print(f"Waiting up to {args.wait_timeout}s for engine readiness (first cold SM121 JIT compile takes ~8-12 min)...")
        t0 = time.monotonic()
        last_log = 0.0
        while time.monotonic() - t0 < args.wait_timeout:
            ok, detail = check_http_ready(args.port, args.host)
            if ok:
                elapsed = int(time.monotonic() - t0)
                print(f"[{elapsed}s] {detail}")
                return

            st0 = run_local(["docker", "inspect", "-f", "{{.State.Running}}", args.container_name]).stdout.strip()
            st1 = run_node1(
                args.node1_ssh,
                f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(args.container_name)}",
            ).stdout.strip()
            if st0 != "true" or st1 != "true":
                tail0 = run_local(["docker", "logs", "--tail", "40", args.container_name])
                tail1 = run_node1(args.node1_ssh, f"docker logs --tail 40 {shlex.quote(args.container_name)}")
                sys.stderr.write(f"\nERROR: Container exited prematurely (Rank0={st0}, Rank1={st1}).\n")
                sys.stderr.write("--- Rank 0 (Node 0) last 40 lines ---\n" + (tail0.stdout or "") + (tail0.stderr or "") + "\n")
                sys.stderr.write("--- Rank 1 (Node 1) last 40 lines ---\n" + (tail1.stdout or "") + (tail1.stderr or "") + "\n")
                sys.exit(1)

            now = time.monotonic()
            if now - last_log >= 20.0:
                print(f"  [{int(now - t0)}s] Initializing weights / CUDA graphs / SM121 kernels...")
                last_log = now
            time.sleep(5)

        sys.exit(f"ERROR: Timed out after {args.wait_timeout}s waiting for engine readiness on port {args.port}")


if __name__ == "__main__":
    main()
