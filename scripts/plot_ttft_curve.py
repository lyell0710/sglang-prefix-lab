#!/usr/bin/env python3
"""单图单结论:TTFT 随共享前缀长度下降,disable-radix 反例臂持平。
用法:plot_ttft_curve.py <derived_csv> <out_png> <title-result-sentence> <model_label>

标题由调用方传"结论句":图自带结论,脱离正文也能被正确解读。同一脚本出
fig1(0.6B,EXP-P03)与 fig2(8B,EXP-P07),模型差异只体现在数据与标签,
图形语言保持一致以支持两图对读(收益天花板随模型规模变化)。
误差条 = 3 seeds 的 p50 std;脚注把 src 文件名 + provenance 首行打进图片,
图不脱离数据谱系(哪份 csv、哪条命令、什么硬件)。
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")   # 无头渲染:服务器无显示环境
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 中文字体显式注册(uming.ttc 为本机可用字体):不设 fallback,缺字直接可见,不静默出豆腐块。
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
# off 臂只画 c1:反例臂一条即可锚定"持平",多画反而喧宾夺主。
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
