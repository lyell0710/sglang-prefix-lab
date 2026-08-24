#!/usr/bin/env python3
"""P04/P08 聚合:bench_groups raw json → derived csv。用法:aggregate_groups.py <EXP-dir> <out_csv> [fname_hint]"""
import glob, json, statistics as st, sys
P, OUT = sys.argv[1], sys.argv[2]
rows = {}
for f in sorted(glob.glob(f"{P}/*_s2026082*.json")):
    parts = f.replace(".json","").split("_"); pol, kind = parts[-3], parts[-2]
    ds = [json.loads(l) for l in open(f) if l.startswith("{")]
    if ds: rows.setdefault((kind, pol), []).append(ds[0])
with open(OUT, "w") as out:
    out.write(f"# provenance: env=sglang-lab cmd=\"python scripts/aggregate_groups.py {P} {OUT}\" source={P}/*_s*.json (3 seeds/cell)\n")
    out.write("tier,policy,rounds,p50_mean_ms,p50_std,p95_mean_ms,p95_std,p99_mean_ms,p99_std,hit_frac_mean,hit_frac_std,dur_s_mean\n")
    for (kind, pol), ds in sorted(rows.items()):
        g = lambda k: [d[k] for d in ds]; m = lambda k: st.mean(g(k)); s = lambda k: st.stdev(g(k))
        out.write(f"{kind},{pol},{len(ds)},{m('ttft_p50_ms'):.1f},{s('ttft_p50_ms'):.1f},{m('ttft_p95_ms'):.1f},{s('ttft_p95_ms'):.1f},{m('ttft_p99_ms'):.1f},{s('ttft_p99_ms'):.1f},{m('hit_fraction'):.4f},{s('hit_fraction'):.4f},{m('duration_s'):.2f}\n")
print("wrote", OUT)
