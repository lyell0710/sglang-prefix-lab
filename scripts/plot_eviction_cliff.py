#!/usr/bin/env python3
"""单图单结论(EXP-P05):LRU 逐出不是斜坡是悬崖——池 < 重用距离即 1.0→0.06 阶跃。
用法:plot_eviction_cliff.py data/derived/exp_p05_eviction_cliff.csv figures/fig3_p05_eviction_cliff.png"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()

src, out = sys.argv[1], sys.argv[2]
prov = ""
rows = []
with open(src) as f:
    for line in f:
        if line.startswith("#"):
            prov = line.strip("# \n"); continue
        break
for row in csv.DictReader(l for l in open(src) if not l.startswith("#")):
    rows.append(row)

# x 轴 = 重用距离 D=8192×(1+cr);series = 池位
dists = sorted({int(r["reuse_distance_tokens"]) for r in rows})
pools = [("8192", "#c0392b", "池 8192(=最小 D,越线即崩)"),
         ("16384", "#0f4c81", "池 16384"),
         ("default", "#1a6fb8", "默认池(≈57 万,远大于 D)")]
cell = {(r["pool_tokens"], int(r["reuse_distance_tokens"])):
        (float(r["hot_hit_frac_mean"]), float(r["hot_hit_frac_std"])) for r in rows}

fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=220)
w = 0.26
for i, (pool, color, label) in enumerate(pools):
    xs, ys, es = [], [], []
    for j, d in enumerate(dists):
        if (pool, d) in cell:
            m, s = cell[(pool, d)]
            xs.append(j + (i - 1) * w); ys.append(m); es.append(s)
    b = ax.bar(xs, ys, width=w, color=color, label=label, yerr=es, capsize=3,
               error_kw={"lw": 1})
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center",
                fontsize=8, color=color)
ax.set_xticks(range(len(dists)))
ax.set_xticklabels([f"{d}\n(cr={d//8192-1})" for d in dists])
ax.set_xlabel("热前缀重用距离 D = 8192×(1+cold_ratio)(token)")
ax.set_ylabel("热前缀命中率(3 seeds mean±std,std=0)")
ax.set_ylim(0, 1.42)
ax.axhline(1.0, color="#999", lw=0.8, ls=":")
ax.set_title("LRU 逐出不是斜坡是悬崖:池 < 重用距离 D 时命中 1.0→0.06 阶跃崩塌\n"
             "(Qwen3-0.6B,EXP-P05,三池 × 四档冷流量,无中间态)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper center", ncol=3, framealpha=0.95)
ax.grid(alpha=0.25, axis="y")
fig.text(0.01, 0.01, f"src: {src.split('/')[-1]} · {prov[:90]} · RTX 4090", fontsize=5, color="#999")
fig.tight_layout()
fig.savefig(out, facecolor="white")
print("saved", out)
