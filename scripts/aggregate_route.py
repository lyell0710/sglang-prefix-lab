#!/usr/bin/env python3
"""P06 聚合:bench_route_pool raw json → derived csv。只聚合有效批次(165910)。"""
import glob, json, statistics as st
rows = {}
for f in sorted(glob.glob("data/raw/EXP-P06/20260824T165910_*_s2026082*.json")):
    parts = f.replace(".json","").split("_")
    pol = parts[-3] + "_" + parts[-2] if parts[-3] in ("round","cache") else None
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
