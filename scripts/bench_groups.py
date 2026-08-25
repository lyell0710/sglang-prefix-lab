#!/usr/bin/env python3
"""EXP-P04/P08 · 分组共享前缀负载:--schedule-policy fcfs vs lpm 的调度放大。

解决什么问题:验证 cache-aware 调度(lpm,最长前缀匹配排序)相对 fcfs 的
真实收益窗口。实测结论必须带定语:0.6B std 档判平、boundary 档 lpm p99 反劣
13%、hit −2.4pp(EXP-P04);8B boundary 档 lpm p50 −62%、hit +17.7pp 但 p99
+64%(EXP-P08)——延迟在分位数间再分配,不是标量优劣。

负载:G 组 × R 请求/组,组内共享 prefix_len 前缀,后缀唯一;全列表 shuffle(seed 固定)
后以 concurrency 并发注入(对抗序:相邻请求大概率不同组)。
为什么 shuffle:按组顺序注入时 fcfs 也天然聚簇同组请求,两策略没有差异空间;
打散后"把同前缀请求重新聚到一起"的能力才真正归属调度器。
机制预期(跑前锁定):lpm 按最长前缀匹配排序聚簇同组请求(schedule_policy.py:373-384),
fcfs 按到达序打散;lpm 命中率与 TTFT 尾部应更优(命中差随 KV 池竞争加剧)。
实测对预期的修正:等待队列 >128 时 lpm 退化回 fcfs(schedule_policy.py:290-294),
排序建立的聚簇假设中途失效——boundary 档(192 req @ c64)即故意跨过该窗口
的边界证据;in-batch 前缀去重的 deprioritize 路径(阈值 32)是另一候选机理。

TTFT 停表口径与 bench_prefix.py 全仓统一:首个 content delta chunk 到达停表。
输出:单行 JSON(逐请求 ttft/cached + 汇总分位数);每策略各起一次 worker,
flush 后注入,3 seeds。

面试点:①hit_possible 分母为什么扣掉 G——每组首请求必 miss(树里尚无该前缀),
理论可命中数 = prefix_len×(N−G),命中率才可跨配置比较;②"开 lpm 总没错"是
伪直觉:负载轻时到达序已足够友好,重过 128 等待窗口后排序反成负资产。
"""
import argparse, asyncio, json, random, time
import aiohttp

def ids_of(n, seed):
    random.seed(seed)
    return random.choices(range(1000, 100000), k=n)   # 避开 special token 区,同 bench_prefix

async def one(session, base, model, ids, out_tokens):
    # messages 仅为 schema 占位;input_ids 直传保证 token 流受控(同 bench_prefix)。
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
                # 停表口径:首个非空 content delta 到达客户端(全仓统一)。
                ttft = (time.perf_counter() - t0) * 1e3
            u = d.get("usage")
            if u and u.get("prompt_tokens_details"):
                cached = u["prompt_tokens_details"].get("cached_tokens")
    return {"ttft_ms": ttft, "cached": cached or 0}   # usage 缺席按 0:保守计 miss

async def run(a):
    # 组内共享前缀、后缀唯一:seed 派生保证跨复跑逐 token 相同。
    reqs = []
    for g in range(a.groups):
        prefix = ids_of(a.prefix_len, a.seed * 100 + g)
        for r in range(a.per_group):
            suffix = ids_of(a.total_len - a.prefix_len, a.seed * 100000 + g * 1000 + r)
            reqs.append((g, prefix + suffix))
    random.seed(a.seed)
    random.shuffle(reqs)                      # 对抗到达序:聚簇能力归属调度器而非注入序
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(a.base_url + "/flush_cache") as r:
            fl = (await r.text())[:20]        # 截断存证:核对 success 即可
        sem = asyncio.Semaphore(a.concurrency)
        async def lim(item):
            async with sem:
                return await one(s, a.base_url, a.model, item[1], a.output_len)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[lim(x) for x in reqs])
        dur = time.perf_counter() - t0
    ttfts = sorted(r["ttft_ms"] for r in res if r["ttft_ms"] is not None)
    n = len(ttfts); pct = lambda p: ttfts[min(n - 1, int(p * n))]   # 索引分位近似,跨 seed 再取 mean±std
    hit_tokens = sum(r["cached"] for r in res)
    possible = a.prefix_len * (len(reqs) - a.groups)   # 每组首请求必 miss:分母扣 G 才是理论上限
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
    ap.add_argument("--groups", type=int, default=8)       # std 档 G8×R8=64 req;boundary 档 G16×R12=192(>128 窗口)
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
