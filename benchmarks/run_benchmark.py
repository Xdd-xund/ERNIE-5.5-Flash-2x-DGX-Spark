#!/usr/bin/env python3
"""
End-to-End Streaming Benchmark & Lossless Quality Gate for `baidu/ERNIE-5.5-Flash-Preview`
on 2x NVIDIA DGX Spark (`2x GB10 SM121`, `2x 128 GiB`, Dual 200GbE ConnectX-7 RoCEv2).

Measures:
  1. Single-stream TTFT, decode tok/s, E2E tok/s, and DSpark speculative acceptance (thinking / code / chat).
  2. Concurrency scaling (C=1, 2, 4, 8 at greedy T=0.0 and sampled T=0.6).
  3. Cold ~7K & ~28K prefill throughput + Warm ~28K prefix-cache hit TTFT.
  4. 4-test lossless quality gate (modular arithmetic, live-executed Python LRU+TTL, strict JSON, parallel tool calls).
"""
from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# Bypass ambient HTTP_PROXY/HTTPS_PROXY so local/LAN cluster benchmarks never hit an external proxy
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPTS = [
    "Write a concise, production-grade Python implementation of an async token-bucket rate limiter using asyncio.Lock, including docstrings and a short usage example.",
    "Derive the closed-form solution of T(n) = 2T(n/2) + n log n using recursion trees and verify the asymptotic bound step by step.",
    "Explain how RoCEv2 Congestion Notification (DCQCN) works across dual ConnectX-7 200GbE links and how PFC deadlock is prevented.",
    "Implement an iterative Tarjan strongly connected components algorithm in Rust with clear comments and O(V + E) complexity analysis.",
    "Design a lock-free single-producer single-consumer ring buffer in C++20 using std::atomic memory_order_acquire and memory_order_release.",
    "Compare Heavily Compressed Attention (HCA) with Multi-head Latent Attention (MLA) in terms of KV cache bytes per token and prefill FLOPs.",
    "Write a SQL query using recursive CTEs and window functions to compute 7-day rolling retention and cohort revenue from an events table.",
    "Prove that every symmetric real matrix has real eigenvalues and orthogonal eigenvectors.",
]


def get_json(base_url: str, path: str) -> dict:
    with _DIRECT_OPENER.open(f"{base_url.rstrip('/')}{path}", timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def get_metrics(base_url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with _DIRECT_OPENER.open(f"{base_url.rstrip('/')}/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", errors="ignore")
        for ln in txt.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$", ln.strip())
            if m:
                k = m.group(1)
                v = float(m.group(3))
                out[k] = out.get(k, 0.0) + v
    except Exception:
        pass
    return out


def spec_delta(m0: dict[str, float], m1: dict[str, float]) -> dict[str, float | int | None]:
    d_acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - m0.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0.0
    )
    d_draft = m1.get("vllm:spec_decode_num_draft_tokens_total", 0.0) - m0.get(
        "vllm:spec_decode_num_draft_tokens_total", 0.0
    )
    d_drafts = m1.get("vllm:spec_decode_num_drafts_total", 0.0) - m0.get(
        "vllm:spec_decode_num_drafts_total", 0.0
    )
    return {
        "dspark_accept_pct": round(100.0 * d_acc / d_draft, 2) if d_draft > 0 else None,
        "dspark_tokens_per_step": round(1.0 + d_acc / d_drafts, 2) if d_drafts > 0 else None,
        "drafts": int(round(d_drafts)),
    }


def sample_cluster_mem(base_url: str, node1_ssh: str | None) -> dict:
    def parse_free(txt: str) -> dict:
        for ln in txt.splitlines():
            parts = ln.split()
            if len(parts) >= 7 and (parts[0].startswith("Mem") or "内存" in parts[0]):
                return {
                    "total_gib": round(int(parts[1]) / (1024**3), 2),
                    "used_gib": round(int(parts[2]) / (1024**3), 2),
                    "free_gib": round(int(parts[3]) / (1024**3), 2),
                    "buff_cache_gib": round(int(parts[5]) / (1024**3), 2),
                    "avail_gib": round(int(parts[6]) / (1024**3), 2),
                }
        return {}

    mem: dict[str, dict] = {}
    parsed = urllib.parse.urlparse(base_url)
    if (parsed.hostname or "127.0.0.1") in ("127.0.0.1", "localhost", "::1"):
        try:
            r0 = subprocess.run(["free", "-b"], text=True, capture_output=True, check=False).stdout
            if r0:
                mem["node0"] = parse_free(r0)
        except FileNotFoundError:
            pass

    if node1_ssh:
        try:
            r1 = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node1_ssh, "free -b"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout
            if r1:
                mem["node1"] = parse_free(r1)
        except FileNotFoundError:
            pass
    return mem


async def sse_chat_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 448,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    tools: list | None = None,
) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if tools is not None:
        payload["tools"] = tools
        payload["stream"] = False
        payload.pop("stream_options", None)

    body = json.dumps(payload).encode("utf-8")

    if not payload.get("stream"):
        def _sync_post() -> tuple[dict, float]:
            t_req = time.perf_counter()
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _DIRECT_OPENER.open(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8")), time.perf_counter() - t_req

        res, elapsed = await asyncio.to_thread(_sync_post)
        msg = res["choices"][0]["message"]
        usage = res.get("usage", {})
        return {
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_content": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
        }

    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")

    def _sync_sse_stream() -> dict:
        t0 = time.perf_counter()
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(host, port, timeout=300)
        first_tok_t: float | None = None
        last_tok_t: float = t0
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        usage: dict = {}

        try:
            conn.request(
                "POST",
                f"{prefix}/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "close",
                },
            )
            resp = conn.getresponse()
            if resp.status != 200:
                err_body = resp.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"HTTP {resp.status} from /v1/chat/completions: {err_body}")

            while True:
                raw = resp.readline()
                if not raw:
                    break
                s = raw.decode("utf-8", errors="ignore").strip()
                if not s.startswith("data:"):
                    continue
                data_str = s[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    ev = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                choices = ev.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    rc = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    ct = delta.get("content") or ""
                    if rc or ct:
                        now = time.perf_counter()
                        if first_tok_t is None:
                            first_tok_t = now
                        last_tok_t = now
                        if rc:
                            reasoning_parts.append(rc)
                        if ct:
                            content_parts.append(ct)
        finally:
            conn.close()

        comp_toks = int(usage.get("completion_tokens") or 0)
        prompt_toks = int(usage.get("prompt_tokens") or 0)
        ttft_ms = round(((first_tok_t or last_tok_t) - t0) * 1000.0, 2)
        decode_s = max(last_tok_t - (first_tok_t or t0), 1e-4)
        decode_tps = round(max(comp_toks - 1, 1) / decode_s, 2)
        e2e_tps = round(comp_toks / max(last_tok_t - t0, 1e-4), 2)
        return {
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tps,
            "e2e_tok_s": e2e_tps,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "elapsed_s": round(last_tok_t - t0, 3),
            "reasoning_text": "".join(reasoning_parts),
            "content_text": "".join(content_parts),
        }

    return await asyncio.to_thread(_sync_sse_stream)


async def run_quality_gate(base_url: str, model: str) -> dict:
    checks: dict[str, dict] = {}

    # 1. Exact modular arithmetic check (7^2026 + 11^2026 mod 100 == 10)
    r_math = await sse_chat_one(
        base_url,
        model,
        "Find the remainder when 7^2026 + 11^2026 is divided by 100. End your response with `FINAL_ANSWER: <integer>`.",
        max_tokens=1024,
        enable_thinking=True,
        temperature=0.0,
        reasoning_effort="low",
    )
    ans_m = re.search(r"FINAL_ANSWER:\s*(\d+)", r_math["content_text"])
    math_ok = bool(ans_m and int(ans_m.group(1)) == 10)
    checks["math_mod100"] = {
        "passed": math_ok,
        "answer": int(ans_m.group(1)) if ans_m else None,
        "decode_tok_s": r_math["decode_tok_s"],
    }

    # 2. Code synthesis + live Python execution (LRUCacheWithTTL)
    r_code = await sse_chat_one(
        base_url,
        model,
        "Implement `class LRUCacheWithTTL` in Python inside a ```python ... ``` code block with:\n"
        "- `__init__(self, capacity: int, default_ttl: float, clock=time.monotonic)`\n"
        "- `get(self, key, default=None)`: returns value if present and `clock() < expiry`, updating LRU order; otherwise evicts if expired and returns `default`.\n"
        "- `put(self, key, value, ttl: float | None = None)`: sets value with `ttl` (or `default_ttl`), updates LRU order, evicts expired keys first before evicting the least-recently-used live key when at `capacity`.\n"
        "- `__len__(self)`: returns count of currently non-expired items.",
        max_tokens=1536,
        enable_thinking=False,
        temperature=0.0,
    )
    code_str = r_code["content_text"].strip()
    m_fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", code_str, flags=re.DOTALL | re.IGNORECASE)
    if m_fence:
        code_str = m_fence.group(1).strip()
    else:
        code_str = re.sub(r"^```[a-zA-Z]*\s*\n", "", code_str)
        code_str = re.sub(r"\n```\s*$", "", code_str)
        code_str = re.sub(r"</?[a-zA-Z0-9_]+>\s*$", "", code_str).strip()

    test_harness = (
        "import time\n"
        + code_str
        + "\nnow = [100.0]\n"
        "c = LRUCacheWithTTL(2, default_ttl=10.0, clock=lambda: now[0])\n"
        "c.put('a', 1); c.put('b', 2, ttl=5.0)\n"
        "assert len(c) == 2 and c.get('a') == 1\n"
        "now[0] = 106.0\n"
        "assert c.get('b') is None and len(c) == 1\n"
        "c.put('c', 3); c.put('d', 4)\n"
        "assert c.get('a') is None and c.get('c') == 3 and c.get('d') == 4\n"
        "print('CODE_OK')\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(test_harness)
        tf_path = tf.name
    try:
        proc = subprocess.run(["python3", tf_path], text=True, capture_output=True, timeout=10)
        code_ok = proc.returncode == 0 and "CODE_OK" in proc.stdout
        code_err = proc.stderr[:200] if proc.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        code_ok = False
        code_err = "Execution timed out after 10s"
    finally:
        Path(tf_path).unlink(missing_ok=True)
    checks["code_lru_ttl"] = {
        "passed": code_ok,
        "decode_tok_s": r_code["decode_tok_s"],
        "stderr": code_err,
    }

    # 3. Strict JSON output without markdown fences
    r_json = await sse_chat_one(
        base_url,
        model,
        'Return ONLY a raw JSON object (no markdown fences, no commentary) with keys "cluster" (string "dual-spark"), "nodes" (integer 2), and "protocols" (array ["rocev2", "nvlink-c2c"]).',
        max_tokens=128,
        enable_thinking=False,
        temperature=0.0,
    )
    raw_j = r_json["content_text"].strip()
    try:
        parsed_j = json.loads(raw_j)
        json_ok = (
            not raw_j.startswith("```")
            and parsed_j.get("cluster") == "dual-spark"
            and parsed_j.get("nodes") == 2
            and parsed_j.get("protocols") == ["rocev2", "nvlink-c2c"]
        )
    except Exception:
        json_ok = False
    checks["strict_json"] = {"passed": json_ok, "raw": raw_j[:160]}

    # 4. Parallel tool calling
    r_tool = await sse_chat_one(
        base_url,
        model,
        "Check GPU temperature on both spark1 and spark2 using get_node_temp.",
        max_tokens=256,
        enable_thinking=True,
        temperature=0.0,
        reasoning_effort="low",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_node_temp",
                    "description": "Query live GPU/SoC temperature in Celsius for a cluster node.",
                    "parameters": {
                        "type": "object",
                        "properties": {"node": {"type": "string"}},
                        "required": ["node"],
                    },
                },
            }
        ],
    )
    tcalls = r_tool.get("tool_calls") or []
    nodes_called: list[str] = []
    for tc in tcalls:
        if tc.get("function", {}).get("name") == "get_node_temp":
            try:
                arg_obj = json.loads(tc["function"].get("arguments") or "{}")
                if isinstance(arg_obj, dict):
                    nodes_called.append(str(arg_obj.get("node", "")))
            except Exception:
                pass
    nodes_called.sort()
    checks["parallel_tool_call"] = {
        "passed": nodes_called == ["spark1", "spark2"],
        "nodes_called": nodes_called,
    }

    passed_cnt = sum(1 for v in checks.values() if v["passed"])
    return {"passed": passed_cnt, "total": len(checks), "all_passed": passed_cnt == len(checks), "checks": checks}


async def run_suite(
    *,
    base_url: str,
    model: str,
    label: str,
    out_path: str,
    verify_quality: bool,
    node1_ssh: str | None,
) -> None:
    models_info = get_json(base_url, "/v1/models")
    print(f"[{label}] Connected to {base_url} -> {model} ({len(models_info.get('data', []))} model)")

    await sse_chat_one(base_url, model, "Reply OK.", max_tokens=32, enable_thinking=False, temperature=0.0)
    await sse_chat_one(base_url, model, "Compute 19 * 23.", max_tokens=64, enable_thinking=True, temperature=0.6, reasoning_effort="low")

    # 1. Single-stream matrix (C=1)
    single_stream = {}
    for name, kw in [
        ("c1_thinking_t06", dict(prompt=PROMPTS[1], max_tokens=512, enable_thinking=True, temperature=0.6, reasoning_effort="low")),
        ("c1_code_t00", dict(prompt=PROMPTS[0], max_tokens=448, enable_thinking=False, temperature=0.0)),
        ("c1_chat_t06", dict(prompt=PROMPTS[2], max_tokens=448, enable_thinking=False, temperature=0.6)),
    ]:
        m0 = get_metrics(base_url)
        res = await sse_chat_one(base_url, model, **kw)
        m1 = get_metrics(base_url)
        res.pop("reasoning_text", None)
        res.pop("content_text", None)
        res.update(spec_delta(m0, m1))
        single_stream[name] = res
        print(
            f"  [{name}] decode={res['decode_tok_s']} tok/s | e2e={res['e2e_tok_s']} tok/s | "
            f"ttft={res['ttft_ms']} ms | step={res['dspark_tokens_per_step']} ({res['dspark_accept_pct']}%)"
        )

    # 2. Concurrency scaling (C=1, 2, 4, 8 at temp=0.0 & temp=0.6)
    scaling_greedy = []
    for c in (1, 2, 4, 8):
        m0 = get_metrics(base_url)
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[sse_chat_one(base_url, model, PROMPTS[i % len(PROMPTS)], max_tokens=448, enable_thinking=False, temperature=0.0) for i in range(c)]
        )
        wall_s = time.perf_counter() - t_start
        m1 = get_metrics(base_url)
        tot_comp = sum(r["completion_tokens"] for r in results)
        sd = spec_delta(m0, m1)
        row = {
            "concurrency": c,
            "wall_s": round(wall_s, 2),
            "total_completion_tokens": tot_comp,
            "aggregate_tok_s": round(tot_comp / max(wall_s, 1e-4), 2),
            "per_stream_decode_tok_s": round(sum(r["decode_tok_s"] for r in results) / c, 2),
            "mean_ttft_ms": round(sum(r["ttft_ms"] for r in results) / c, 2),
            **sd,
        }
        print(
            f"  [Greedy C={c}] agg={row['aggregate_tok_s']} tok/s | stream={row['per_stream_decode_tok_s']} tok/s | "
            f"ttft={row['mean_ttft_ms']} ms | step={row['dspark_tokens_per_step']} ({row['dspark_accept_pct']}%)"
        )
        scaling_greedy.append(row)

    scaling_sampled = []
    for c in (2, 4, 8):
        m0 = get_metrics(base_url)
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[sse_chat_one(base_url, model, PROMPTS[i % len(PROMPTS)], max_tokens=448, enable_thinking=False, temperature=0.6) for i in range(c)]
        )
        wall_s = time.perf_counter() - t_start
        m1 = get_metrics(base_url)
        tot_comp = sum(r["completion_tokens"] for r in results)
        sd = spec_delta(m0, m1)
        row = {
            "concurrency": c,
            "wall_s": round(wall_s, 2),
            "total_completion_tokens": tot_comp,
            "aggregate_tok_s": round(tot_comp / max(wall_s, 1e-4), 2),
            "per_stream_decode_tok_s": round(sum(r["decode_tok_s"] for r in results) / c, 2),
            "mean_ttft_ms": round(sum(r["ttft_ms"] for r in results) / c, 2),
            **sd,
        }
        print(
            f"  [Sampled t=0.6 C={c}] agg={row['aggregate_tok_s']} tok/s | stream={row['per_stream_decode_tok_s']} tok/s | "
            f"ttft={row['mean_ttft_ms']} ms | step={row['dspark_tokens_per_step']} ({row['dspark_accept_pct']}%)"
        )
        scaling_sampled.append(row)

    # 3. Cold & warm prefill scaling (~7K and ~28K tokens, guaranteed cold via UUID nonce)
    ctx_rows = []
    base_block = "System architecture note: Dual DGX Spark connects two GB10 Grace-Blackwell superchips via dual 200GbE ConnectX-7 RoCEv2 links. "
    nonce_7k = f"[Session-{uuid.uuid4()}]\n"
    prompt_7k = nonce_7k + (base_block * (4096 // 20)) + "\nSummarize the interconnect topology above in two concise sentences."
    r7 = await sse_chat_one(base_url, model, prompt_7k, max_tokens=96, enable_thinking=False, temperature=0.0)
    ctx_rows.append({
        "mode": "cold_7k",
        "prompt_tokens": r7["prompt_tokens"],
        "ttft_ms": r7["ttft_ms"],
        "prefill_tok_s": round(r7["prompt_tokens"] / max(r7["ttft_ms"] / 1000.0, 1e-4), 1),
        "decode_tok_s": r7["decode_tok_s"],
    })
    print(f"  [Cold ~7K: {r7['prompt_tokens']}t] TTFT={r7['ttft_ms']} ms ({ctx_rows[-1]['prefill_tok_s']} tok/s) | decode={r7['decode_tok_s']} tok/s")

    nonce_28k = f"[Session-{uuid.uuid4()}]\n"
    prompt_28k = nonce_28k + (base_block * (16384 // 20)) + "\nSummarize the interconnect topology above in two concise sentences."
    r28_cold = await sse_chat_one(base_url, model, prompt_28k, max_tokens=96, enable_thinking=False, temperature=0.0)
    ctx_rows.append({
        "mode": "cold_28k",
        "prompt_tokens": r28_cold["prompt_tokens"],
        "ttft_ms": r28_cold["ttft_ms"],
        "prefill_tok_s": round(r28_cold["prompt_tokens"] / max(r28_cold["ttft_ms"] / 1000.0, 1e-4), 1),
        "decode_tok_s": r28_cold["decode_tok_s"],
    })
    print(f"  [Cold ~28K: {r28_cold['prompt_tokens']}t] TTFT={r28_cold['ttft_ms']} ms ({ctx_rows[-1]['prefill_tok_s']} tok/s) | decode={r28_cold['decode_tok_s']} tok/s")

    r28_warm = await sse_chat_one(base_url, model, prompt_28k + " Be specific.", max_tokens=96, enable_thinking=False, temperature=0.0)
    ctx_rows.append({
        "mode": "warm_28k_prefix_hit",
        "prompt_tokens": r28_warm["prompt_tokens"],
        "ttft_ms": r28_warm["ttft_ms"],
        "effective_tok_s": round(r28_warm["prompt_tokens"] / max(r28_warm["ttft_ms"] / 1000.0, 1e-4), 1),
        "decode_tok_s": r28_warm["decode_tok_s"],
    })
    print(f"  [Warm ~28K: {r28_warm['prompt_tokens']}t] TTFT={r28_warm['ttft_ms']} ms ({ctx_rows[-1]['effective_tok_s']} tok/s) | decode={r28_warm['decode_tok_s']} tok/s")

    quality = None
    if verify_quality:
        print("  Running quality regression gate...")
        quality = await run_quality_gate(base_url, model)
        print(f"  [Quality Gate] {quality['passed']}/{quality['total']} passed -> {quality['checks']}")

    report = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "memory_after_bench": sample_cluster_mem(base_url, node1_ssh),
        "single_stream": single_stream,
        "concurrency_scaling_greedy": scaling_greedy,
        "concurrency_scaling_sampled": scaling_sampled,
        "context_prefill": ctx_rows,
        "quality_gate": quality,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{label}] Saved report -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark & Quality Gate for ERNIE-5.5-Flash-Preview on Dual DGX Spark")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8898"))
    ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "baidu/ERNIE-5.5-Flash-Preview"))
    ap.add_argument("--label", default="local_run")
    ap.add_argument("--out", default="benchmarks/results/local_run.json")
    ap.add_argument("--node1-ssh", default=os.environ.get("NODE1_SSH", ""))
    ap.add_argument("--verify-quality", action="store_true", help="Run 4-test lossless quality gate")
    args = ap.parse_args()
    asyncio.run(
        run_suite(
            base_url=args.base_url,
            model=args.model,
            label=args.label,
            out_path=args.out,
            verify_quality=args.verify_quality,
            node1_ssh=args.node1_ssh or None,
        )
    )


if __name__ == "__main__":
    main()
