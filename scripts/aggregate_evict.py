#!/usr/bin/env python3
"""P05 聚合:bench_evict raw json → derived csv(附重用距离模型列)。

按 (池位, cold_ratio) 归组 3 seeds;池位从文件名 hint 反推(smallpool→8192,
midpool16k→16384,否则 default≈16 万(161671,EXP-P01 启动日志))——raw 命名即协议编码。
reuse_distance_tokens 列 = 8192×(1+cr),即 D = hot_count×total_len×(1+cr):
把"悬崖位置可由模型预测"直接写进表,读表即可对照 池≥D ⇔ hit=1.0。
EXP-P05 锚:hit 呈 1.0 与 ~1/16 量级残余的两态(16384@cr4 为 0.125),seed 间 std=0。
"""
import glob, json, statistics as st, sys
rows = {}
for f in sorted(glob.glob("data/raw/EXP-P05/*_s2026082*.json")):   # _s<seed> 命名锁定正式轮,试跑不混入
    d = [json.loads(l) for l in open(f) if l.startswith("{")][0]
    pool = "8192" if "smallpool" in f else ("16384" if "midpool16k" in f else "default")
    rows.setdefault((pool, d["cold_ratio"]), []).append(d)
with open("data/derived/exp_p05_eviction_cliff.csv", "w") as out:
    out.write('# provenance: env=sglang-lab cmd="python scripts/aggregate_evict.py" source=data/raw/EXP-P05/*.json (3 seeds/cell)\n')
    out.write("pool_tokens,cold_ratio,reuse_distance_tokens,hot_hit_frac_mean,hot_hit_frac_std,full_hits_mean_of_16,evicted_tokens_endrun_mean\n")
    for (pool, cr), ds in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1])):
        h = [d["hot_hit_frac_mean"] for d in ds]; fh = [d["hot_full_hit_count"] for d in ds]
        ev = [d["evicted_tokens_total"] or 0 for d in ds]   # 默认池从未逐出时 counter 不曝光(None)→ 按 0 记
        out.write(f"{pool},{cr},{8192*(1+cr)},{st.mean(h):.4f},{st.stdev(h) if len(h)>1 else 0:.4f},{st.mean(fh):.1f},{st.mean(ev):.0f}\n")
print("wrote exp_p05_eviction_cliff.csv")
