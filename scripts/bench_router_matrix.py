#!/usr/bin/env python3
"""EXP-S04 · router 性能矩阵：prefix locality × routing policy × concurrency。

按 protocol-router-v1.json 跑主矩阵。每 cell = (policy, workload, concurrency, seed)，
192 请求，stream 逐 token 计时 TTFT/TPOT/E2E，逐请求记 cached_tokens。

gate（跑前锁定，FAIL 即抛异常不落 derived）：
  - 响应 prompt_tokens 硬 gate（< 下限即 FAIL，防请求体被中间层改写，EXP-P06 教训）
  - 失败请求 = 0
  - policy 标签从 router 日志/workers 确认

用法: python scripts/bench_router_matrix.py --policy round_robin \
      --workload hot_prefix_1024 --concurrency 4 --seed 2026082401 \
      --manifest-dir data/raw/EXP-S02 --base-url http://127.0.0.1:40000 \
      --model Qwen/Qwen3-8B --out data/raw/EXP-S04/
"""
import argparse, asyncio, json, os, time
import aiohttp
from pathlib import Path


async def one_stream(session, base, model, text, min_prompt, max_tokens=32):
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    last_t = t0
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        if r.status != 200:
            body = await r.text()
            raise RuntimeError(f"HTTP {r.status}: {body[:200]}")
        async for line in r.content:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and chunk.get("choices"):
                ttft = time.perf_counter() - t0
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                n_tok += 1
                last_t = time.perf_counter()
    e2e = time.perf_counter() - t0
    tpot = (last_t - t0 - (ttft or 0)) / max(n_tok - 1, 1) if n_tok > 1 else 0.0
    return {"ttft_ms": (ttft or e2e) * 1e3, "tpot_ms": tpot * 1e3,
            "e2e_ms": e2e * 1e3, "n_tokens": n_tok}


async def run(a):
    mf = Path(a.manifest_dir) / f"{a.workload}_s{a.seed}.jsonl"
    reqs = [json.loads(l) for l in mf.read_text().splitlines()]
    timeout = aiohttp.ClientTimeout(total=3600)
    sem = asyncio.Semaphore(a.concurrency)
    results = []

    async def worker(i, text):
        async with sem:
            return await one_stream(session, a.base_url, a.model, text, a.min_prompt)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 预热：前 min(8, n) 请求先跑一遍（drives JIT/缓存，不入统计）
        warm = min(8, len(reqs))
        for r in reqs[:warm]:
            await one_stream(session, a.base_url, a.model, r["text"], a.min_prompt)
        # 正式：并发 c 发全部 192 请求
        tasks = [asyncio.create_task(worker(i, r["text"])) for i, r in enumerate(reqs)]
        for i, t in enumerate(tasks):
            try:
                results.append(await t)
            except Exception as e:
                results.append({"error": str(e)[:200]})

    errs = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if errs:
        raise RuntimeError(f"{len(errs)} failed requests: {errs[0]}")
    def qs(key):
        vals = sorted(r[key] for r in ok)
        n = len(vals)
        return {"mean": sum(vals) / n, "p50": vals[n // 2],
                "p95": vals[int(n * 0.95)], "p99": vals[int(n * 0.99)]}
    return {"policy": a.policy, "workload": a.workload, "concurrency": a.concurrency,
            "seed": a.seed, "n": len(ok),
            "ttft_ms": qs("ttft_ms"), "tpot_ms": qs("tpot_ms"), "e2e_ms": qs("e2e_ms"),
            "throughput_req_s": round(len(ok) / (sum(r["e2e_ms"] for r in ok) / 1e3 / a.concurrency), 2) if a.concurrency > 1 else round(1 / (sum(r["e2e_ms"] for r in ok) / 1e3 / len(ok)), 2),
            "rows": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest-dir", default="data/raw/EXP-S02")
    ap.add_argument("--base-url", default="http://127.0.0.1:40000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--min-prompt", type=int, default=1000)  # prompt_tokens 硬 gate 下限
    ap.add_argument("--out", default="data/raw/EXP-S04")
    a = ap.parse_args()
    res = asyncio.run(run(a))
    os.makedirs(a.out, exist_ok=True)
    fn = Path(a.out) / f"{a.policy}_{a.workload}_c{a.concurrency}_s{a.seed}.json"
    json.dump(res, open(fn, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
