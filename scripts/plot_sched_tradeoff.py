#!/usr/bin/env python3
"""单图单结论(EXP-P08):8B 积压档 lpm 赢 p50 输 p99——延迟在分位数间再分配。
用法:plot_sched_tradeoff.py data/derived/exp_p08_8b_fcfs_vs_lpm.csv figures/fig4_p08_sched_tradeoff.png"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()

src, out = sys.argv[1], sys.argv[2]
prov = ""
with open(src) as f:
    for line in f:
        if line.startswith("#"):
            prov = line.strip("# \n"); continue
        break
data = {}
for row in csv.DictReader(l for l in open(src) if not l.startswith("#")):
    if row["tier"] != "boundary":
        continue  # 单图单结论:只画积压档(std 档两策略无可区分,见 EXP-P08 §6)
    data[row["policy"]] = row

metrics = [("p50", "TTFT p50"), ("p99", "TTFT p99")]
policies = [("fcfs", "#c0392b", "fcfs(基线)"), ("lpm", "#1a6fb8", "lpm")]
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=220)
h = 0.32
ys, labels = [], []
for gi, (m, mlabel) in enumerate(metrics):
    for pi, (pol, color, plabel) in enumerate(policies):
        y = gi * 1.0 + (pi - 0.5) * h
        v = float(data[pol][f"{m}_mean_ms"]); e = float(data[pol][f"{m}_std"])
        ax.barh(y, v, height=h * 0.92, color=color, xerr=e, capsize=3,
                error_kw={"lw": 1}, label=plabel if gi == 0 else None)
        ax.text(v + e + 250, y, f"{v:,.0f}", va="center", fontsize=9, color=color)
    ys.append(gi * 1.0); labels.append(mlabel)
ax.set_yticks(ys); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("TTFT(ms,3 seeds mean±std,G16×R12 @ 并发 64)")
ax.set_xlim(0, 19500)
ax.set_title("8B 积压档:lpm p50 -62% 但 p99 +64%——延迟在分位数间再分配,不是标量优劣\n"
             "(Qwen3-8B,EXP-P08 boundary 档;std 档两策略无可区分)", fontsize=10)
hf, hl = float(data["fcfs"]["hit_frac_mean"]), float(data["lpm"]["hit_frac_mean"])
ax.text(0.985, 0.72, f"命中率:fcfs {hf:.3f} → lpm {hl:.3f}(+{(hl-hf)*100:.1f}pp)\n"
        "机理:lpm 聚簇同前缀请求换命中,\n代价是排序靠后的组整组延后",
        transform=ax.transAxes, ha="right", fontsize=8, color="#333",
        bbox=dict(fc="#f5f5f5", ec="#999", lw=0.5))
ax.legend(fontsize=8, loc="center right")
ax.grid(alpha=0.25, axis="x")
fig.text(0.01, 0.01, f"src: {src.split('/')[-1]} · {prov[:90]} · 2026-08-24", fontsize=5, color="#999")
fig.tight_layout()
fig.savefig(out, facecolor="white")
print("saved", out)
