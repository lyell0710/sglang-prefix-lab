#!/usr/bin/env python3
"""EXP-P06 · 路由 × 池容量:"cache-aware 路由 = 扩大有效缓存容量"的实证检验。

预测(由 EXP-P05 重用距离模型跑前锁定):双 worker 各限池 8192 token,
热工作集 6×2048=12288 > 单池、< 双池之和。
- round_robin:同一前缀交替打到两卡,每卡看到全部 6 前缀轮转 → 每卡重用距离
  ≈ 6×2048=12288 > 8192 → 双卡都 thrash,hot_hit → ~0
- cache_aware:每前缀被钉在一张卡,每卡 ~3 前缀 → 距离 ≈ 6144 < 8192 → hit → ~1

实测双双证伪(EXP-P06,3 seeds 全格 std=0;预注册文本按史料保留):
- rr@hot6 反而全命中(1.0000):轮转周期 6 与 worker 数 2 奇偶对齐,严格轮询
  意外成为完美分片(每卡 3 前缀,距离 ~6450 < 8192);hot5 奇数对照打破整除
  关系后 rr 立即崩(0.0020)——是巧合不是能力。
- cache_aware 反而全崩(0.0020)且流量 100/0(61799/0):串行 c=1 下负载恒 0,
  失衡回退(64/1.5 阈值)永不触发,冷启动把全部前缀钉到同一张卡,~12900 的
  工作集塞进 8192 单池 → thrash。亲和只有在 tenant 分散多卡时才等效扩容。

通过 router(40000)发请求;逐请求 cached_tokens;两 worker /metrics 差分佐证
流量落点(路由决策是黑盒,由 worker 侧 prompt_tokens_total 计数器反推)。
V2 修正(EXP-P06 首轮 16:53 全批作废换来的教训):router 严格按 OpenAI schema
重序列化,**丢弃 input_ids 扩展字段**——全部请求曾静默退化成 ~10 token(靠
cached=8 与 worker 增量 135 反推发现)。修正三件套:
  ① 负载改文本形态(tokenizer decode 的随机 token 文本)——这也正是
     cache_aware 近似树匹配的对象,顺带对齐了被测机制;
  ② 响应 prompt_tokens 硬 gate(低于下限即抛异常 FAIL),杜绝任何中间层改写
     请求体后实验静默继续;
  ③ 每臂结束校验 router selection 指标的 policy 标签(svc.sh 旧身份校验曾被
     router 的 setproctitle 改名绕过,第一支 router 存活跨臂,cache_aware 臂
     实际仍在跑 rr)。

面试点:①"测量链路上每一跳都可能改写你的请求"——input_ids 被 router 丢弃
是静默的,唯一可靠防线是对响应内硬计数(prompt_tokens)设 gate;②rr 的全命中
依赖热集数与副本数的整除关系,换个热集数即崩,不可依赖;③cache-aware 不等于
扩容:前缀→副本映射的质量(分散且稳定)才是本质。
"""
import argparse, asyncio, json, random, time
import aiohttp

def ids_of(n, seed):
    random.seed(seed); return random.choices(range(1000, 100000), k=n)

_TOK = None
def text_of(n, seed):
    """随机 token 解码成文本(bench_serving 同法);重编码长度≈n 有漂移,实际以响应 prompt_tokens 为准。"""
    return _TOK.decode(ids_of(n, seed))

async def one(session, base, model, text, out_tokens=8, min_prompt=1000):
    # 文本形态负载:router 会按 OpenAI schema 重序列化,只有 messages 能存活。
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": False}
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        d = await r.json()
    u = d.get("usage", {}) or {}
    pt = u.get("prompt_tokens") or 0
    if pt < min_prompt:                       # 硬 gate:防请求体被中间层静默改写(首轮教训)
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
            # 直接对各 worker flush:radix tree 在 worker 侧,router 不代理管理端点。
            async with s.post(w + "/flush_cache") as r:
                await r.text()
        await asyncio.sleep(1)               # 等 flush 落定再抓基线,防差分含清树前残留
        before = [await grab(s, w + "/metrics") for w in a.worker_urls]
        for h in hots:                                  # 预热经 router:预热流量参与 tenant 分配,冷启动行为正是被测对象
            await one(s, a.base_url, a.model, h + text_of(32, random.randint(1, 9999999)),
                      min_prompt=a.prefix_len // 2)
        hot_cached = []; prompt_lens = []
        t0 = time.perf_counter()
        for i in range(a.rounds_per_hot * a.hot_count):
            hot = hots[i % a.hot_count]                 # 轮转:重用距离模型的 D 构造(同 EXP-P05)
            # gate 下限取 prefix_len//2(=768 @1536):文本重编码长度有 ± 漂移,
            # gate 只需挡"payload 被丢弃退化到 ~10 token"的量级失败,不追求精确等长。
            c, pt = await one(s, a.base_url, a.model,
                              hot + " " + text_of(a.total_len - a.prefix_len, a.seed * 7 + i),
                              min_prompt=a.prefix_len // 2)
            hot_cached.append(c); prompt_lens.append(pt)
        dur = time.perf_counter() - t0
        after = [await grab(s, w + "/metrics") for w in a.worker_urls]
        loads = await grab(s, a.base_url + "/get_loads")
    def counter(txt, name):
        # Prometheus 文本解析:指标名后可能带 {labels} 也可能裸名,两种前缀都接受。
        for line in txt.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                try: return float(line.split()[-1])
                except ValueError: pass
        return 0.0
    # 差分而非绝对值:worker 计数器跨臂累计,只有 before/after 差才归属本臂。
    per_worker_prompts = [
        counter(after[i], "sglang:prompt_tokens_total") - counter(before[i], "sglang:prompt_tokens_total")
        for i in range(len(a.worker_urls))]
    est_prefix = min(prompt_lens) if prompt_lens else a.prefix_len   # 命中上限保守估计(重编码漂移),仅诊断用
    hit_frac = [min(1.0, c / a.prefix_len) for c in hot_cached]      # 钳 1.0:实际公共前缀可能略长于名义 prefix_len
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
    ap.add_argument("--base-url", default="http://127.0.0.1:40000")   # router 端口;worker 直连仅用于 flush/metrics
    ap.add_argument("--worker-urls", nargs="+",
                    default=["http://127.0.0.1:28000", "http://127.0.0.1:28001"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hot-count", type=int, default=6)      # 6=偶数主臂;对照臂 5(奇)打破与 2 卡的整除关系
    ap.add_argument("--prefix-len", type=int, default=1536)
    ap.add_argument("--total-len", type=int, default=2048)
    ap.add_argument("--rounds-per-hot", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--tag", default="")                     # 本臂期望的 policy 名,随 raw 存证(另有指标标签 gate)
    ap.add_argument("--tokenizer-dir", required=True)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a)), ensure_ascii=False))

if __name__ == "__main__":
    main()
