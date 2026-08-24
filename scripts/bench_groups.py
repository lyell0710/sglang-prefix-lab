#!/usr/bin/env python3
"""EXP-P04 · 分组共享前缀负载:--schedule-policy fcfs vs lpm 的调度放大。

负载:G 组 × R 请求/组,组内共享 prefix_len 前缀,后缀唯一;全列表 shuffle(seed 固定)
后以 concurrency 并发注入(对抗序:相邻请求大概率不同组)。
机制预期:lpm 按最长前缀匹配排序聚簇同组请求(schedule_policy.py:373-384),
fcfs 按到达序打散;lpm 命中率与 TTFT 尾部应更优(命中差随 KV 池竞争加剧)。
输出:单行 JSON(逐请求 ttft/cached + 汇总)。
"""
import argparse, asyncio, json, random, time
import aiohttp

def ids_of(n, seed):
    random.seed(seed)
    return random.choices(range(1000, 100000), k=n)

async def one(session, base, model, ids, out_tokens):
    payload = {"model": model, "input_ids": ids,
               "messages": [{"role": "user", "content": "x"}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; cached = None
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        async for raw in r.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            body = line[5:].strip()
            if body == "[DONE]": break
            try: d = json.loads(body)
            except json.JSONDecodeError: continue
            ch = d.get("choices") or []
            if ttft is None and ch and (ch[0].get("delta") or {}).get("content"):
                ttft = (time.perf_counter() - t0) * 1e3
            u = d.get("usage")
            if u and u.get("prompt_tokens_details"):
                cached = u["prompt_tokens_details"].get("cached_tokens")
    return {"ttft_ms": ttft, "cached": cached or 0}

async def run(a):
    reqs = []
    for g in range(a.groups):
        prefix = ids_of(a.prefix_len, a.seed * 100 + g)
        for r in range(a.per_group):
            suffix = ids_of(a.total_len - a.prefix_len, a.seed * 100000 + g * 1000 + r)
            reqs.append((g, prefix + suffix))
    random.seed(a.seed)
    random.shuffle(reqs)                      # 对抗到达序
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(a.base_url + "/flush_cache") as r:
            fl = (await r.text())[:20]
        sem = asyncio.Semaphore(a.concurrency)
        async def lim(item):
            async with sem:
                return await one(s, a.base_url, a.model, item[1], a.output_len)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[lim(x) for x in reqs])
        dur = time.perf_counter() - t0
    ttfts = sorted(r["ttft_ms"] for r in res if r["ttft_ms"] is not None)
    n = len(ttfts); pct = lambda p: ttfts[min(n - 1, int(p * n))]
    hit_tokens = sum(r["cached"] for r in res)
    possible = a.prefix_len * (len(reqs) - a.groups)   # 每组首请求必 miss
    return {"groups": a.groups, "per_group": a.per_group,
            "prefix_len": a.prefix_len, "total_len": a.total_len,
            "concurrency": a.concurrency, "seed": a.seed, "flush": fl,
            "completed": n, "duration_s": round(dur, 2),
            "ttft_p50_ms": round(pct(0.5), 2), "ttft_p95_ms": round(pct(0.95), 2),
            "ttft_p99_ms": round(pct(0.99), 2), "ttft_mean_ms": round(sum(ttfts)/n, 2),
            "hit_tokens": hit_tokens, "hit_possible": possible,
            "hit_fraction": round(hit_tokens / possible, 4) if possible else None,
            "ttft_ms": [round(t, 2) for t in ttfts]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--prefix-len", type=int, default=1536)
    ap.add_argument("--total-len", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260824)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a)), ensure_ascii=False))

if __name__ == "__main__":
    main()
