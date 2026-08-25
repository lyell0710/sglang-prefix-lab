#!/usr/bin/env python3
"""P06 聚合:bench_route_pool raw json → derived csv。只聚合有效批次(165910)。

为什么把时间戳写死进 glob:首轮(16:53)整批作废——router 丢弃 input_ids
扩展字段致负载静默退化(EXP-P06 §7),作废 raw 按红线保留原地但永不入聚合,
锁时间戳防未来误混。config 轴 hot5_odd/hot6_even 即奇偶对照:rr 的"全命中"
是否为轮转周期与副本数的整除巧合,由 hot5 打破整除后立即崩塌坐实。
worker0/1_traffic 为 prompt_tokens_total 差分均值:cache_aware 的 100/0 单卡
集中(61799/0)由该列直接可见。3 seeds 全格 std=0。
"""
import glob, json, statistics as st
rows = {}
for f in sorted(glob.glob("data/raw/EXP-P06/20260824T165910_*_s2026082*.json")):
    parts = f.replace(".json","").split("_")
    pol = parts[-3] + "_" + parts[-2] if parts[-3] in ("round","cache") else None   # 历史残留启发式,实际以下两行判定
    cfg = "hot5_odd" if "_hot5_" in f else "hot6_even"
    d = [json.loads(l) for l in open(f) if l.startswith("{")][0]
    pol = "round_robin" if "round_robin" in f else "cache_aware"
    rows.setdefault((cfg, pol), []).append(d)
with open("data/derived/exp_p06_routing_pool.csv", "w") as out:
    out.write('# provenance: env=sglang-lab cmd="python scripts/aggregate_route.py" source=data/raw/EXP-P06/20260824T165910_*.json (3 seeds/cell, std=0)\n')
    out.write("config,policy,hot_hit_frac_mean,full_hits_mean,n_hot,worker0_traffic_mean,worker1_traffic_mean\n")
    for (cfg, pol), ds in sorted(rows.items()):
        h = [d["hot_hit_frac_mean"] for d in ds]
        w0 = [d["per_worker_prompt_tokens_delta"][0] for d in ds]
        w1 = [d["per_worker_prompt_tokens_delta"][1] for d in ds]
        out.write(f"{cfg},{pol},{st.mean(h):.4f},{st.mean([d['hot_full_hit_count'] for d in ds]):.1f},{ds[0]['n_hot_requests']},{st.mean(w0):.0f},{st.mean(w1):.0f}\n")
print("wrote exp_p06_routing_pool.csv")
