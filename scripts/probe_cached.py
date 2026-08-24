#!/usr/bin/env python3
"""EXP-P01 探针:确定性 + radix 首证(同 prompt 双发,第二次 cached_tokens>0)。"""
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
    same = rounds[0]["content"] == rounds[1]["content"]
    hit = (rounds[1]["cached_tokens"] or 0) > 0
    out = {"models_ok": bool(models.get("data")), "deterministic": same,
           "second_send_cache_hit": hit, "rounds":
           [{k: v for k, v in r.items() if k != "content"} for r in rounds],
           "passed": bool(models.get("data")) and same and hit}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
