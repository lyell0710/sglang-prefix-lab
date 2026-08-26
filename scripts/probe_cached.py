#!/usr/bin/env python3
"""EXP-P01（env 与单 worker smoke）探针:确定性 + radix 命中首证(同 prompt 双发,第二发 cached_tokens>0)。

解决什么问题:在跑任何收益曲线之前,用最小实验证明三个前提同时成立:
① server 可用(/v1/models 有数据);② temperature=0 双发 content 逐字符相等
——确定性是后续一切对比实验的地基;③ 第二发 cached_tokens>0——radix 缓存
真实生效。三者合取才 PASS,退出码 0/1 可直接作 gate 串进实验流程。

实测锚(EXP-P01,单轮确定性验证):第二发 cached=1324/1325,恰为 n−1——
调度器把前缀匹配上限压到 input_len−1,必须至少重算 1 个 token 否则没有
logits 可采样(docs/theory/01 §2.2);hit_rate 0.9992,flashinfer 后端。

面试点:cached 上限为什么是 n−1 不是 n——不是 off-by-one bug,是采样的硬
要求;实测值精确落在源码读出的上限,是"理论→测量"闭环的最小样本。
"""
import argparse, json, time, urllib.request

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    a = ap.parse_args()
    with urllib.request.urlopen(f"{a.base_url}/v1/models", timeout=30) as r:
        models = json.loads(r.read())
    # 长 user 文本放大共享前缀(模板头 + 该文本在第二发应命中)
    text = "请逐条复述以下清单,不要添加内容:" + "".join(
        f"第{i}条,内容编号{i*7};" for i in range(1, 120))
    payload = {"model": a.model,
               "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": 32, "stream": False}
    rounds = []
    for i in range(2):
        d, dt = post(f"{a.base_url}/v1/chat/completions", payload)
        u = d.get("usage", {}) or {}
        det = u.get("prompt_tokens_details") or {}
        rounds.append({"latency_s": round(dt, 4),
                       "prompt_tokens": u.get("prompt_tokens"),
                       "cached_tokens": det.get("cached_tokens"),
                       "content": d["choices"][0]["message"]["content"]})
    same = rounds[0]["content"] == rounds[1]["content"]   # 逐字符相等:确定性判据
    hit = (rounds[1]["cached_tokens"] or 0) > 0           # 首证只要求 >0;精确 n-1 对账见记录
    out = {"models_ok": bool(models.get("data")), "deterministic": same,
           "second_send_cache_hit": hit, "rounds":
           [{k: v for k, v in r.items() if k != "content"} for r in rounds],
           "passed": bool(models.get("data")) and same and hit}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
