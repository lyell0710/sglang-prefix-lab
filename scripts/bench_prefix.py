#!/usr/bin/env python3
"""EXP-P03/P07 · 命中收益曲线:TTFT vs 共享前缀长度(input_ids 直传,流式停表)。

解决什么问题:量化"前缀命中到底省多少 TTFT",并证明收益确实来自前缀复用
——disable-radix 起服的 off 臂与 prefix=0 点必须持平,否则归因不成立。

臂协议(每个测量点三段,顺序即因果链):
  flush_cache(须 success)→ 预热臂:发 1 条 [prefix] 请求把前缀灌进树(不计时)
  → 计时臂:N 条 [prefix + 唯一后缀] 请求,流式,首 content chunk 停表 = TTFT。
  --no-warm 跳过预热(冷对照);前缀长度 0 = 全唯一(反例臂)。
  为什么 flush:清掉上一测量点的树,点与点零残留,否则 prefix_lens 的扫描
  顺序会污染结果;为什么预热:把"树里已有前缀"设为前置条件,计时臂测的是
  纯命中收益,而不是"首个请求替大家种树"的混合态;为什么后缀唯一:保证可
  命中的只有前缀段,server 报的 cached_tokens 应恰=prefix_len,可逐请求硬
  校验(聚合侧 aggregate_p03.py 的 cached_ok_all 列)。

TTFT 停表口径(全仓统一):t0=POST 发出前;停表=流中第一个带 delta.content
的 chunk 到达客户端时刻。不取 HTTP 首字节(那只是 SSE 响应头,不含 token),
也不取 server 侧直方图(要的是含排队+prefill+首 token 解码的用户可感知延迟)。

接口契约:--prefix-lens 逗号列表逐点扫描;stdout 每点一行 JSONL(汇总分位数
+ 逐请求 ttft_ms 数组 + cached_tokens 数组);重跑以时间前缀写新文件,不覆盖 raw。

实测锚:Qwen3-8B prefix 1792/2048 时 TTFT p50 228.4→52.9 ms(并发 1,−77%)、
1068.3→234.5 ms(并发 8,−78%),EXP-P07;0.6B 同协议仅 −36%/−63%(EXP-P03)
——收益正比于被跳过的 prefill 在 TTFT 中的占比,模型越大天花板越高。

面试点:①为什么 input_ids 直传而不发文本——绕过 chat template 渲染,token
序列完全受控,cached_tokens 才能与 prefix_len 逐 token 对账(EXP-P02:模板/
thinking 开关都会改 token 流);②usage 为何随流取(stream_options.include_usage)
——停表与命中数在同一次响应内闭合,"快了多少"与"命中了多少"同源,归因
不靠事后查 /metrics 推测。
"""
import argparse, asyncio, json, random, sys, time
import aiohttp

def build_ids(tok, n):
    # 1000..100000:避开词表低位(字节/控制符)与高位 special token 区,
    # 保证序列对 tokenizer 与引擎都是"普通 token",不触发任何模板特判。
    vocab = [i for i in range(1000, 100000)]
    return random.choices(vocab, k=n)

async def one_request(session, url, model, ids, out_tokens):
    # messages 仅为 OpenAI schema 占位;server 端 input_ids 扩展字段优先生效。
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
                # 停表口径:首个非空 content delta(自动跳过 role-only 首 chunk)。
                ttft = (time.perf_counter() - t0) * 1e3
            u = d.get("usage")
            if u and u.get("prompt_tokens_details"):
                # 命中数随流取:与停表同一响应内闭合,归因不依赖事后指标。
                cached = u["prompt_tokens_details"].get("cached_tokens")
    e2e = (time.perf_counter() - t0) * 1e3
    return {"ttft_ms": ttft, "e2e_ms": e2e, "cached_tokens": cached}

async def flush(session, url):
    async with session.post(url + "/flush_cache") as r:
        return (await r.text())[:40]   # 截断存证:raw 中核对含 success 即可

async def run_point(args, tok_seed, prefix_len, concurrency, warm):
    # 可复现负载:prefix 只由 tok_seed 决定(点内所有请求共享同一前缀);
    # 每条后缀用独立派生 seed——彼此不同(唯一性)且跨复跑逐 token 相同。
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
            # 预热:max_tokens=1 把成本压到最低;+8 随机尾 token 使预热请求
            # 区别于计时请求(尾巴唯一、不会被后续命中),prefix 整段随请求
            # 完成插进 radix tree。
            await one_request(s, args.base_url, args.model, prefix + build_ids(None, 8), 1)
        sem = asyncio.Semaphore(concurrency)
        async def lim(ids):
            # 客户端信号量限流即"并发"的定义:在途请求数上限,而非 server batch。
            async with sem:
                return await one_request(s, args.base_url, args.model, ids, args.output_len)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[lim(r) for r in reqs])
        dur = time.perf_counter() - t0
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    ttfts_sorted = sorted(ttfts)
    # 简单索引分位(int(p*n) 钳到 n-1):16 样本下的近似;结论一律再跨 3 seeds
    # 取 mean±std,分位定义的细微偏差不影响臂间相对比较。
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
    ap.add_argument("--total-len", type=int, default=2048)   # 定总长扫前缀:变量只有 prefix_len
    ap.add_argument("--prefix-lens", default="0,512,1024,1536,1792")   # 1792=2048 的 87.5%,记录口径
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=16)   # 16 req/点 × 3 seeds,与 records 口径一致
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args()
    for pl in [int(x) for x in a.prefix_lens.split(",")]:
        row = asyncio.run(run_point(a, a.seed, pl, a.concurrency, not a.no_warm))
        print(json.dumps(row, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
