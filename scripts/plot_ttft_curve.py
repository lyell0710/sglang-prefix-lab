#!/usr/bin/env python3
"""单图单结论:TTFT 随共享前缀长度下降,disable-radix 反例臂持平。
用法:plot_ttft_curve.py <derived_csv> <out_png> <title-result-sentence> <model_label>"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()

src, out, title, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
series = {}
prov = ""
with open(src) as f:
    for line in f:
        if line.startswith("#"): prov = line.strip("# \n"); continue
        break
for row in csv.DictReader(l for l in open(src) if not l.startswith("#")):
    key = (row["arm"], row["concurrency"])
    series.setdefault(key, []).append(
        (int(row["prefix_len"]), float(row["ttft_p50_mean_ms"]), float(row["ttft_p50_std_ms"])))
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=220)
style = {("on","1"): ("#1a6fb8","o","radix on · 并发1"),
         ("on","8"): ("#0f4c81","s","radix on · 并发8"),
         ("off","1"): ("#c0392b","^","disable-radix · 并发1(反例臂)")}
for key, pts in sorted(series.items()):
    pts.sort()
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; es=[p[2] for p in pts]
    c,m,lab = style.get(key, ("#777","x",str(key)))
    ax.errorbar(xs, ys, yerr=es, color=c, marker=m, capsize=3, lw=1.6, label=lab)
ax.set_xlabel("共享前缀长度(token,总长 2048)")
ax.set_ylabel("TTFT p50(ms,3 seeds mean±std)")
ax.set_title(f"{title}\n({model})", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.text(0.01, 0.01, f"src: {src.split('/')[-1]} · {prov[:80]} · RTX 4090", fontsize=5, color="#999")
fig.tight_layout()
fig.savefig(out)
print("saved", out)
