#!/usr/bin/env python3
"""EXP-P03 · 命中收益曲线:TTFT vs 共享前缀长度(input_ids 直传,流式停表)。

协议(每个测量点):
  flush_cache(须 success)→ 预热臂:发 1 条 [prefix] 请求把前缀灌进树(不计时)
  → 计时臂:N 条 [prefix + 唯一后缀] 请求,流式,首 chunk 停表 = TTFT。
  --no-warm 跳过预热(冷对照);前缀长度 0 = 全唯一(反例臂)。
输出 JSONL:每行一个测量点,含逐请求 ttft_ms 数组与 server 侧 cached_tokens。
"""
import argparse, asyncio, json, random, sys, time
import aiohttp

def build_ids(tok, n):
    vocab = [i for i in range(1000, 100000)]
    return random.choices(vocab, k=n)

async def one_request(session, url, model, ids, out_tokens):
    payload = {"model": model, "input_ids": ids,
               "messages": [{"role": "user", "content": "x"}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None; cached = None
    async with session.post(url + "/v1/chat/completions", json=payload) as r:
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
    e2e = (time.perf_counter() - t0) * 1e3
    return {"ttft_ms": ttft, "e2e_ms": e2e, "cached_tokens": cached}

async def flush(session, url):
    async with session.post(url + "/flush_cache") as r:
        return (await r.text())[:40]

async def run_point(args, tok_seed, prefix_len, concurrency, warm):
    random.seed(tok_seed)
    prefix = build_ids(None, prefix_len) if prefix_len else []
    reqs = []
    for i in range(args.num_requests):
        random.seed(tok_seed * 1000 + i + 1)
        suffix = build_ids(None, args.total_len - prefix_len)
        reqs.append(prefix + suffix)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        fl = await flush(s, args.base_url)
        if warm and prefix_len:
            await one_request(s, args.base_url, args.model, prefix + build_ids(None, 8), 1)
        sem = asyncio.Semaphore(concurrency)
        async def lim(ids):
            async with sem:
                return await one_request(s, args.base_url, args.model, ids, args.output_len)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[lim(r) for r in reqs])
        dur = time.perf_counter() - t0
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    ttfts_sorted = sorted(ttfts)
    pct = lambda p: ttfts_sorted[min(len(ttfts_sorted)-1, int(p*len(ttfts_sorted)))] if ttfts_sorted else None
    return {"prefix_len": prefix_len, "total_len": args.total_len,
            "concurrency": concurrency, "warm": warm, "seed": tok_seed,
            "num_requests": args.num_requests, "flush": fl,
            "completed": len(ttfts), "duration_s": round(dur, 3),
            "ttft_ms_mean": round(sum(ttfts)/len(ttfts), 2) if ttfts else None,
            "ttft_ms_p50": round(pct(0.5), 2) if ttfts else None,
            "ttft_ms_p95": round(pct(0.95), 2) if ttfts else None,
            "cached_tokens": [r["cached_tokens"] for r in results],
            "ttft_ms": [round(t, 2) for t in ttfts]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--total-len", type=int, default=2048)
    ap.add_argument("--prefix-lens", default="0,512,1024,1536,1792")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args()
    for pl in [int(x) for x in a.prefix_lens.split(",")]:
        row = asyncio.run(run_point(a, a.seed, pl, a.concurrency, not a.no_warm))
        print(json.dumps(row, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
