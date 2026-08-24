#!/usr/bin/env python3
"""EXP-P05 · 逐出压力:小 KV 池下,冷流量把热前缀冲出缓存的退化曲线。

协议:worker 以 --max-total-tokens 限池。H 个热前缀(prefix_len)先各预热一次;
然后按 [热请求 ×1,冷请求 ×cold_ratio] 交替注入(串行,c=1,隔离排队效应),
热请求轮转热前缀。观测:热请求的 cached/prefix_len 命中分数随 cold_ratio 的退化。
输出单行 JSON(逐热请求 cached 序列 + 汇总)。
"""
import argparse, asyncio, json, random, time
import aiohttp

def ids_of(n, seed):
    random.seed(seed); return random.choices(range(1000, 100000), k=n)

async def one(session, base, model, ids, out_tokens=8):
    payload = {"model": model, "input_ids": ids,
               "messages": [{"role": "user", "content": "x"}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": False}
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        d = await r.json()
    det = (d.get("usage", {}) or {}).get("prompt_tokens_details") or {}
    return det.get("cached_tokens") or 0

async def run(a):
    hots = [ids_of(a.prefix_len, a.seed * 100 + h) for h in range(a.hot_count)]
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(a.base_url + "/flush_cache") as r:
            fl = (await r.text())[:20]
        for h in hots:                       # 预热全部热前缀
            await one(s, a.base_url, a.model, h + ids_of(32, random.randint(1, 9999999)))
        hot_cached = []
        t0 = time.perf_counter()
        for i in range(a.hot_requests):
            hot = hots[i % a.hot_count]
            c = await one(s, a.base_url, a.model, hot + ids_of(a.total_len - a.prefix_len, a.seed * 7 + i))
            hot_cached.append(c)
            for j in range(a.cold_ratio):    # 冷流量:全唯一
                await one(s, a.base_url, a.model, ids_of(a.total_len, a.seed * 31 + i * 100 + j))
        dur = time.perf_counter() - t0
        async with s.get(a.base_url + "/metrics") as r:
            met = await r.text()
    evicted = None
    for line in met.splitlines():
        if line.startswith("sglang:evicted_tokens_total"):
            evicted = float(line.split()[-1])
    hit_frac = [c / a.prefix_len for c in hot_cached]
    return {"hot_count": a.hot_count, "prefix_len": a.prefix_len,
            "total_len": a.total_len, "cold_ratio": a.cold_ratio,
            "hot_requests": a.hot_requests, "seed": a.seed, "flush": fl,
            "duration_s": round(dur, 2),
            "hot_hit_frac_mean": round(sum(hit_frac) / len(hit_frac), 4),
            "hot_full_hit_count": sum(1 for c in hot_cached if c >= a.prefix_len - 1),
            "evicted_tokens_total": evicted,
            "hot_cached": hot_cached}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hot-count", type=int, default=4)
    ap.add_argument("--prefix-len", type=int, default=1536)
    ap.add_argument("--total-len", type=int, default=2048)
    ap.add_argument("--cold-ratio", type=int, default=0)
    ap.add_argument("--hot-requests", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260824)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a)), ensure_ascii=False))

if __name__ == "__main__":
    main()
