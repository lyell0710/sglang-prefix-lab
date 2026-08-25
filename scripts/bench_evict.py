#!/usr/bin/env python3
"""EXP-P05 · 逐出压力:小 KV 池下,冷流量把热前缀冲出缓存的退化测量。

解决什么问题:找出前缀缓存失效的容量边界。结论修正了预注册的"退化曲线"
用词:LRU 逐出是悬崖不是斜坡——热命中率只有 1.0 与 0.0625(=1/16,残余即
预热后的首个热请求)两个稳态,池 < 重用距离即阶跃崩塌,三池 × 四档冷流量
仅 16384@cr4 见 0.125 的边界残余,3 seeds 间 std=0(EXP-P05)。

重用距离构造(本实验的核心设计):同一热前缀两次被访问之间注入的 token 量
  D = hot_count × total_len × (1+cold_ratio) = 4×2048×(1+cr) = 8192×(1+cr)
hot_count×total_len 特意取 8192 = 最小池位:cr=0 时 D 恰压在池边界上,
cold_ratio 每 +1 把 D 线性外推一个池位。LRU 命中 ⇔ 池 ≥ D:8192 池 cr=0 保
命中、cr≥1 崩;16384 池 cr=4(D=40960)崩;默认池(≈16 万(161671,EXP-P01 启动日志))全保。

协议:worker 以 --max-total-tokens 限池。H 个热前缀(prefix_len)先各预热一次;
然后按 [热请求 ×1,冷请求 ×cold_ratio] 交替注入(串行,c=1,隔离排队效应:
观测量是命中率不是延迟,并发还会让在途请求的 lock_ref 保护扰动逐出顺序,
机理不再干净),热请求轮转热前缀。观测:热请求的 cached/prefix_len 命中分数
随 cold_ratio 的退化;另抓 /metrics 的 evicted_tokens_total 佐证逐出确实发生
(小池随 cr 单调升 33.9→66.3 万;默认池全程无逐出)。
输出单行 JSON(逐热请求 cached 序列 + 汇总)。

面试点:①轮转访问 + LRU 是经典最坏搭配(循环工作集病态):池略小于工作集时,
每个热前缀总在被再次访问之前恰好被逐出,所以命中不是按比例衰减而是归零——
与 CPU cache 的 LRU thrashing 同构,radix cache 没有魔法,就是带引用计数的
LRU;②工程含义:容量规划按热前缀重用距离配池,不是按热集大小;冷流量占比
把 D 线性推过边界,是一阶变量。
"""
import argparse, asyncio, json, random, time
import aiohttp

def ids_of(n, seed):
    random.seed(seed); return random.choices(range(1000, 100000), k=n)   # 同 bench_prefix:避开 special token 区

async def one(session, base, model, ids, out_tokens=8):
    # 非流式:本实验不停表,只取 usage.cached_tokens,解析最简。
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
        for h in hots:                       # 预热全部热前缀入树(随机尾 token 唯一,不会被后续命中)
            await one(s, a.base_url, a.model, h + ids_of(32, random.randint(1, 9999999)))
        hot_cached = []
        t0 = time.perf_counter()
        for i in range(a.hot_requests):
            hot = hots[i % a.hot_count]      # 轮转访问:重用距离模型 D 的构造前提
            c = await one(s, a.base_url, a.model, hot + ids_of(a.total_len - a.prefix_len, a.seed * 7 + i))
            hot_cached.append(c)
            for j in range(a.cold_ratio):    # 冷流量:全唯一序列,零命中,只消耗池容量、推远热前缀重用距离
                await one(s, a.base_url, a.model, ids_of(a.total_len, a.seed * 31 + i * 100 + j))
        dur = time.perf_counter() - t0
        async with s.get(a.base_url + "/metrics") as r:
            met = await r.text()
    evicted = None
    for line in met.splitlines():
        if line.startswith("sglang:evicted_tokens_total"):
            evicted = float(line.split()[-1])
    # 从未逐出时该 counter 不曝光 → evicted 保持 None,聚合侧按 0 记(EXP-P05 §7)。
    hit_frac = [c / a.prefix_len for c in hot_cached]
    return {"hot_count": a.hot_count, "prefix_len": a.prefix_len,
            "total_len": a.total_len, "cold_ratio": a.cold_ratio,
            "hot_requests": a.hot_requests, "seed": a.seed, "flush": fl,
            "duration_s": round(dur, 2),
            "hot_hit_frac_mean": round(sum(hit_frac) / len(hit_frac), 4),
            # ≥ prefix_len-1:留 1 token 容差(匹配上限/对齐的 off-by-one 边界),不把满命中误判为 miss
            "hot_full_hit_count": sum(1 for c in hot_cached if c >= a.prefix_len - 1),
            "evicted_tokens_total": evicted,
            "hot_cached": hot_cached}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hot-count", type=int, default=4)      # 4×2048=8192:热工作集恰=最小池位
    ap.add_argument("--prefix-len", type=int, default=1536)
    ap.add_argument("--total-len", type=int, default=2048)
    ap.add_argument("--cold-ratio", type=int, default=0)     # 实验扫 {0,1,2,4}:D=8192×(1+cr)
    ap.add_argument("--hot-requests", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260824)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a)), ensure_ascii=False))

if __name__ == "__main__":
    main()
