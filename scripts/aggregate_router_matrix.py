#!/usr/bin/env python3
"""EXP-S04 矩阵聚合：54 个 cell JSON → derived CSV（按 policy×workload×concurrency 聚合 3 seeds）。

输出 data/raw/EXP-S04/derived_matrix.csv：
  每行 = (policy, workload, concurrency)，TTFT/TPOT/E2E 的 p50 取 3 seeds mean，
  并给 round_robin→cache_aware 的相对变化。
"""
import json, os, statistics
from pathlib import Path


def load(d):
    cells = {}
    for fn in Path(d).glob("*.json"):
        r = json.load(open(fn))
        if "error" in r or "ttft_ms" not in r:
            continue
        key = (r["policy"], r["workload"], r["concurrency"])
        cells.setdefault(key, []).append(r)
    return cells


def main():
    d = "data/raw/EXP-S04"
    cells = load(d)
    rows = []
    for key in sorted(cells.keys()):
        pol, wl, c = key
        seeds = cells[key]
        def agg(m):
            return round(statistics.mean(s[m]["p50"] for s in seeds), 2)
        rows.append({"policy": pol, "workload": wl, "concurrency": c,
                     "n_seeds": len(seeds),
                     "ttft_p50_ms": agg("ttft_ms"), "tpot_p50_ms": agg("tpot_ms"),
                     "e2e_p50_ms": agg("e2e_ms"),
                     "throughput_req_s": round(statistics.mean(s["throughput_req_s"] for s in seeds), 2)})
    # 相对变化：cache_aware vs round_robin（同 workload × concurrency）
    rr = {(r["workload"], r["concurrency"]): r for r in rows if r["policy"] == "round_robin"}
    ca = {(r["workload"], r["concurrency"]): r for r in rows if r["policy"] == "cache_aware"}
    for k in sorted(rr.keys()):
        if k in ca:
            wl, c = k
            dt = ca[k]["ttft_p50_ms"] - rr[k]["ttft_p50_ms"]
            pct = dt / rr[k]["ttft_p50_ms"] * 100 if rr[k]["ttft_p50_ms"] else 0
            ca[k]["ttft_delta_pct_vs_rr"] = round(pct, 1)
            rr[k]["ttft_delta_pct_vs_rr"] = 0.0
    out = Path(d) / "derived_matrix.csv"
    with open(out, "w") as f:
        f.write("policy,workload,concurrency,n_seeds,ttft_p50_ms,tpot_p50_ms,e2e_p50_ms,throughput_req_s,ttft_delta_pct_vs_rr\n")
        for r in rows:
            f.write(f"{r['policy']},{r['workload']},{r['concurrency']},{r['n_seeds']},"
                    f"{r['ttft_p50_ms']},{r['tpot_p50_ms']},{r['e2e_p50_ms']},"
                    f"{r['throughput_req_s']},{r.get('ttft_delta_pct_vs_rr','')}\n")
    print(f"written {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
