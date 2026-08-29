#!/usr/bin/env python3
"""EXP-S02 · 单 worker reference 采集（parity probe 的确定性参照）。

对每个 manifest 的 probe 集（前 N 个请求）采单 worker 在 temperature=0 下的
确定性输出，存 JSON + SHA256。S04 的 parity gate 用它证明 A/B 两臂的 worker
输出与单 worker reference 逐 token 一致（即 router 不改变 worker 行为）。

用法: python scripts/gen_router_reference.py --manifest-dir data/raw/EXP-S02 \
      --base-url http://127.0.0.1:28000 --model Qwen/Qwen3-8B --probe 8
"""
import argparse, asyncio, hashlib, json, os
import aiohttp
from pathlib import Path


async def one(session, base, model, text, max_tokens=32):
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": max_tokens, "stream": False}
    async with session.post(base + "/v1/chat/completions", json=payload) as r:
        d = await r.json()
    content = d["choices"][0]["message"]["content"]
    u = d.get("usage", {}) or {}
    det = u.get("prompt_tokens_details") or {}
    return content, u.get("prompt_tokens"), det.get("cached_tokens")


async def run(a):
    mdir = Path(a.manifest_dir)
    files = sorted(mdir.glob("*.jsonl"))
    timeout = aiohttp.ClientTimeout(total=3600)
    out = {}
    async with aiohttp.ClientSession(timeout=timeout) as s:
        for fn in files:
            reqs = [json.loads(l) for l in fn.read_text().splitlines()][:a.probe]
            # 串行采：reference 要确定性，串行避免并发调度影响
            refs = []
            for i, r in enumerate(reqs):
                content, pt, cached = await one(s, a.base_url, a.model, r["text"])
                refs.append({"request_id": r["request_id"],
                             "content": content,
                             "prompt_tokens": pt, "cached_tokens": cached})
            body = json.dumps(refs, ensure_ascii=False, sort_keys=True)
            out[fn.name] = {"sha256": hashlib.sha256(body.encode()).hexdigest(),
                            "refs": refs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="data/raw/EXP-S02")
    ap.add_argument("--base-url", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--probe", type=int, default=8)
    ap.add_argument("--out", default="data/raw/EXP-S02/reference.json")
    a = ap.parse_args()
    res = asyncio.run(run(a))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # 落盘：refs 里 content 是长文本，摘要用 sha256，正文存文件
    summary = {k: {"sha256": v["sha256"], "n_probe": len(v["refs"])}
               for k, v in res.items()}
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "full": res}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
