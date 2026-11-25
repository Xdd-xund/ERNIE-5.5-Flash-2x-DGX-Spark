#!/usr/bin/env python3
"""Unit and integrity tests for the open-source ERNIE-5.5 Dual DGX Spark repository."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class RepoIntegrityTest(unittest.TestCase):
    def test_release_hygiene_passes(self) -> None:
        hygiene = load_module("verify_release_hygiene", "scripts/verify_release_hygiene.py")
        errors = hygiene.verify_repo(REPO_ROOT)
        self.assertEqual(errors, [], f"Hygiene check errors: {errors}")

    def test_required_files_exist(self) -> None:
        required = [
            "README.md",
            "README_CN.md",
            "LICENSE",
            "THIRD-PARTY-NOTICES.md",
            "CHANGELOG.md",
            ".gitignore",
            ".dockerignore",
            ".env.example",
            "Dockerfile",
            "patches/patch_gb10_ernie55.py",
            "scripts/check_cluster_prereqs.sh",
            "scripts/prepare_weights.sh",
            "scripts/build_image.sh",
            "scripts/launch_cluster.py",
            "scripts/verify_release_hygiene.py",
            "benchmarks/run_benchmark.py",
            "benchmarks/results/r0_v010_baseline.json",
            "benchmarks/results/r2_v030_run1.json",
            "benchmarks/results/r2_v030_warm_prefix.json",
            "docs/ARCHITECTURE_AND_PATCHES.md",
            "docs/TROUBLESHOOTING.md",
        ]
        for rel in required:
            self.assertTrue((REPO_ROOT / rel).is_file(), f"Missing required file: {rel}")

    def test_dockerfile_uses_public_base_image_and_pinned_commit(self) -> None:
        df = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0", df)
        self.assertIn("FROM ${BASE_IMAGE}", df)
        self.assertIn("ba322a5634f91d3e6f96ae711df72ddaf3c38b6e", df)
        self.assertIn("--index-strategy unsafe-best-match", df)
        self.assertIn("COPY patches/patch_gb10_ernie55.py /opt/vLLM-ERNIE/patch_gb10_ernie55.py", df)
        self.assertNotIn("FROM local/", df)

    def test_launch_cluster_dotenv_inline_comments_empty_fallback_and_jit_caches(self) -> None:
        launcher = load_module("launch_cluster", "scripts/launch_cluster.py")
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tf:
            tf.write(
                "NODE0_IP=192.168.100.1 # head node\n"
                "NODE1_IP=192.168.100.2\n"
                "NODE1_SSH=dgx@192.168.100.2\n"
                "FABRIC_IFNAME=enp1s0f1np1 # port 1\n"
                "RDMA_HCAS=rocep1s0f1,roceP2p1s0f1\n"
                "NODE1_FABRIC_IFNAME=\n"
                "MODEL_DIR=/srv/models/ernie55\n"
                "NODE1_MODEL_DIR=\n"
                "PORT=8900 # custom port\n"
                "GPU_MEMORY_UTILIZATION=0.85\n"
            )
            env_path = tf.name
        try:
            args = launcher.parse_args(["print-cmd", "--env-file", env_path])
            self.assertEqual(args.node0_ip, "192.168.100.1")
            self.assertEqual(args.node1_ip, "192.168.100.2")
            self.assertEqual(args.fabric_ifname, "enp1s0f1np1")
            self.assertEqual(args.node1_fabric_ifname, "enp1s0f1np1")
            self.assertEqual(args.node1_model_dir, "/srv/models/ernie55")
            self.assertEqual(args.rdma_hcas, "rocep1s0f1,roceP2p1s0f1")
            self.assertEqual(args.port, 8900)
            self.assertAlmostEqual(args.gpu_memory_utilization, 0.85)

            cmd0 = launcher.build_docker_cmd(
                rank=0,
                host_ip=args.node0_ip,
                master_ip=args.node0_ip,
                master_port=args.master_port,
                fabric_ifname=args.fabric_ifname,
                fabric_prefix_len=args.fabric_prefix_len,
                rdma_hcas=args.rdma_hcas,
                ib_gid_index=args.ib_gid_index,
                container_name=args.container_name,
                model_dir=args.model_dir,
                cache_dir=args.cache_dir,
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
                enable_dspark=True,
                num_spec_tokens=7,
                dspark_draft_topk=0,
                adaptive_verification=False,
                enable_flashindexer=True,
            )
            cmd1 = launcher.build_docker_cmd(
                rank=1,
                host_ip=args.node1_ip,
                master_ip=args.node0_ip,
                master_port=args.master_port,
                fabric_ifname=args.node1_fabric_ifname,
                fabric_prefix_len=args.fabric_prefix_len,
                rdma_hcas=args.node1_rdma_hcas,
                ib_gid_index=args.ib_gid_index,
                container_name=args.container_name,
                model_dir=args.node1_model_dir,
                cache_dir=args.node1_cache_dir,
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
                enable_dspark=True,
                num_spec_tokens=7,
                dspark_draft_topk=0,
                adaptive_verification=False,
                enable_flashindexer=True,
            )
            joined0 = " ".join(cmd0)
            self.assertIn("NCCL_PROTO=LL,Simple", joined0)
            self.assertIn("NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1", joined0)
            self.assertIn("FLASHINFER_WORKSPACE_BASE=/root/.cache/vllm/flashinfer", joined0)
            self.assertIn("DG_JIT_CACHE_DIR=/root/.cache/vllm/deep_gemm", joined0)
            self.assertIn("XDG_CACHE_HOME=/root/.cache/vllm/xdg", joined0)
            self.assertNotIn("--headless", cmd0)
            self.assertIn("--headless", cmd1)
            self.assertNotIn("dspark_draft_topk", joined0)
        finally:
            Path(env_path).unlink(missing_ok=True)

    def test_prereq_script_mem_awk_emits_trailing_newline(self) -> None:
        script_txt = (REPO_ROOT / "scripts/check_cluster_prereqs.sh").read_text(encoding="utf-8")
        self.assertIn(r'printf \"%.1f %.1f\n\"', script_txt)
        proc = subprocess.run(
            ["bash", "-c", 'set -euo pipefail; read -r a b < <(printf "%.1f %.1f\\n" 121.7 110.2); [[ "$a" == "121.7" && "$b" == "110.2" ]]'],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)

    def test_benchmark_reports_show_expected_gains_and_quality_gate(self) -> None:
        r0 = json.loads((REPO_ROOT / "benchmarks/results/r0_v010_baseline.json").read_text(encoding="utf-8"))
        r2_1 = json.loads((REPO_ROOT / "benchmarks/results/r2_v030_run1.json").read_text(encoding="utf-8"))
        r2_w = json.loads((REPO_ROOT / "benchmarks/results/r2_v030_warm_prefix.json").read_text(encoding="utf-8"))
        for rpt in (r0, r2_1, r2_w):
            self.assertTrue(rpt["quality_gate"]["all_passed"])
            self.assertEqual(rpt["quality_gate"]["passed"], 4)
        self.assertLess(r2_w["context_prefill"][2]["ttft_ms"], 600.0)
        self.assertGreater(r2_1["concurrency_scaling_sampled"][-1]["aggregate_tok_s"], 115.0)


if __name__ == "__main__":
    unittest.main()
