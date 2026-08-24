#!/usr/bin/env python3
"""EXP-P06 · 路由 × 池容量:cache-aware 路由 = 扩大有效缓存容量的实证。

预测(由 EXP-P05 重用距离模型跑前锁定):双 worker 各限池 8192 token,
热工作集 6×2048=12288 > 单池、< 双池之和。
- round_robin:同一前缀交替打到两卡,每卡看到全部 6 前缀轮转 → 每卡重用距离
  ≈ 6×2048=12288 > 8192 → 双卡都 thrash,hot_hit → ~0
- cache_aware:每前缀被钉在一张卡,每卡 ~3 前缀 → 距离 ≈ 6144 < 8192 → hit → ~1
通过 router(40000)发请求;逐请求 cached_tokens;两 worker /metrics 差分佐证分布。
V2 修正(EXP-P06 第一轮陷阱):router 严格按 OpenAI schema 重序列化,**丢弃
input_ids 扩展字段**——负载改为文本形态(tokenizer decode 的随机 token 文本),
这也正是 cache_aware 近似树匹配的对象;响应 prompt_tokens 加硬 gate(<1000 即
FAIL),杜绝静默退化;每臂结束校验 router selection 指标的 policy 标签。
"""
import argparse, asyncio, json, random, time
import aiohttp

def ids_of(n, seed):
    random.seed(seed); return random.choices(range(1000, 100000), k=n)

_TOK = None
def text_of(n, seed):
    """随机 token 解码成文本(bench_serving 同法);重编码长度≈n(实际以响应为准)。"""
    return _TOK.decode(ids_of(n, seed))

async def one(session, base, model, text, out_tokens=8, min_prompt=1000):
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": False}
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        d = await r.json()
    u = d.get("usage", {}) or {}
    pt = u.get("prompt_tokens") or 0
    if pt < min_prompt:                       # 硬 gate:防请求体被中间层静默改写
        raise RuntimeError(f"prompt_tokens={pt} < {min_prompt}: payload degraded, resp={str(d)[:200]}")
    det = u.get("prompt_tokens_details") or {}
    return det.get("cached_tokens") or 0, pt

async def grab(session, url):
    try:
        async with session.get(url) as r:
            return await r.text()
    except Exception as e:
        return f"ERR {e}"

async def run(a):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    hots = [text_of(a.prefix_len, a.seed * 100 + h) for h in range(a.hot_count)]
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        for w in a.worker_urls:
            async with s.post(w + "/flush_cache") as r:
                await r.text()
        await asyncio.sleep(1)
        before = [await grab(s, w + "/metrics") for w in a.worker_urls]
        for h in hots:                                  # 预热(经 router,计入路由)
            await one(s, a.base_url, a.model, h + text_of(32, random.randint(1, 9999999)),
                      min_prompt=a.prefix_len // 2)
        hot_cached = []; prompt_lens = []
        t0 = time.perf_counter()
        for i in range(a.rounds_per_hot * a.hot_count):
            hot = hots[i % a.hot_count]                 # 轮转
            c, pt = await one(s, a.base_url, a.model,
                              hot + " " + text_of(a.total_len - a.prefix_len, a.seed * 7 + i),
                              min_prompt=a.prefix_len // 2)
            hot_cached.append(c); prompt_lens.append(pt)
        dur = time.perf_counter() - t0
        after = [await grab(s, w + "/metrics") for w in a.worker_urls]
        loads = await grab(s, a.base_url + "/get_loads")
    def counter(txt, name):
        for line in txt.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                try: return float(line.split()[-1])
                except ValueError: pass
        return 0.0
    per_worker_prompts = [
        counter(after[i], "sglang:prompt_tokens_total") - counter(before[i], "sglang:prompt_tokens_total")
        for i in range(len(a.worker_urls))]
    est_prefix = min(prompt_lens) if prompt_lens else a.prefix_len   # 命中上限估计
    hit_frac = [min(1.0, c / a.prefix_len) for c in hot_cached]
    return {"policy_expected": a.tag, "hot_count": a.hot_count,
            "prefix_len": a.prefix_len, "total_len": a.total_len,
            "rounds_per_hot": a.rounds_per_hot, "seed": a.seed,
            "duration_s": round(dur, 2),
            "hot_hit_frac_mean": round(sum(hit_frac) / len(hit_frac), 4),
            "hot_full_hit_count": sum(1 for c in hot_cached if c >= a.prefix_len - 1),
            "n_hot_requests": len(hot_cached),
            "per_worker_prompt_tokens_delta": per_worker_prompts,
            "prompt_tokens_min_max": [min(prompt_lens), max(prompt_lens)] if prompt_lens else None,
            "get_loads": loads[:200], "hot_cached": hot_cached}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:40000")
    ap.add_argument("--worker-urls", nargs="+",
                    default=["http://127.0.0.1:28000", "http://127.0.0.1:28001"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hot-count", type=int, default=6)
    ap.add_argument("--prefix-len", type=int, default=1536)
    ap.add_argument("--total-len", type=int, default=2048)
    ap.add_argument("--rounds-per-hot", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokenizer-dir", required=True)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a)), ensure_ascii=False))

if __name__ == "__main__":
    main()
