#!/usr/bin/env python3
"""EXP-S02 · router 矩阵 manifest 生成（不可变、可复现）。

按 config/protocol-router-v1.json 生成 3 workload × 3 seeds 的请求 manifest。
每个 (workload, seed) 一份 JSONL，192 requests，token 序列确定生成（固定 seed）
→ 同输入跨 A/B 两臂 SHA256 相同，是「A/B 使用同一 manifest」gate 的前提。

workload 语义（docs/PLAN_router_matrix.md，总长 2048 token）：
  unique_control   192 个全独立 2048-token 请求（无共享前缀）
  hot_prefix_1024  共享 1024-token hot prefix + 1024-token 随机后缀
  hot_prefix_1792  共享 1792-token hot prefix + 256-token 随机后缀

manifest 字段：request_id / workload / seed / prefix_tokens / suffix_tokens /
  text（tokenizer decode 的文本，router 只认 messages 文本形态，EXP-P06 教训）。

用法: python scripts/gen_router_manifest.py --out data/raw/EXP-S02/
"""
import argparse, hashlib, json, os, random
from pathlib import Path


def ids_of(n, seed):
    random.seed(seed)
    return random.choices(range(1000, 100000), k=n)


def gen_manifest(workload, seed, n_req, tok):
    total = 2048
    if workload == "unique_control":
        reqs = []
        for i in range(n_req):
            ids = ids_of(total, seed * 100000 + i)
            reqs.append({"request_id": f"{workload}_s{seed}_r{i}",
                         "workload": workload, "seed": seed,
                         "prefix_tokens": 0, "suffix_tokens": total,
                         "text": tok.decode(ids)})
    else:
        plen = 1024 if workload == "hot_prefix_1024" else 1792
        slen = total - plen
        prefix_ids = ids_of(plen, seed * 99991 + 7)  # 前缀固定(共享)
        prefix_text = tok.decode(prefix_ids)
        reqs = []
        for i in range(n_req):
            suffix_ids = ids_of(slen, seed * 100000 + i)
            reqs.append({"request_id": f"{workload}_s{seed}_r{i}",
                         "workload": workload, "seed": seed,
                         "prefix_tokens": plen, "suffix_tokens": slen,
                         "text": prefix_text + " " + tok.decode(suffix_ids)})
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/EXP-S02")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--requests", type=int, default=192)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026082401, 2026082402, 2026082403])
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    os.makedirs(a.out, exist_ok=True)
    workloads = ["unique_control", "hot_prefix_1024", "hot_prefix_1792"]
    summary = {}
    for w in workloads:
        for s in a.seeds:
            reqs = gen_manifest(w, s, a.requests, tok)
            body = "\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n"
            fn = Path(a.out) / f"{w}_s{s}.jsonl"
            fn.write_text(body)
            sha = hashlib.sha256(body.encode()).hexdigest()
            summary[f"{w}_s{s}"] = {"file": fn.name, "sha256": sha, "n": len(reqs)}
            # 验证 token 长度（decode 后再 encode 会有漂移，记录名义长度 + 抽样重编码）
            r0 = reqs[0]
            renc = len(tok.encode(r0["text"]))
            summary[f"{w}_s{s}"]["req0_rencoded_tokens"] = renc
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
