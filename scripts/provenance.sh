#!/usr/bin/env bash
# provenance 公共函数:每个 raw 文件的首行血统标记由 prov_line 生成。
# 设计:数字的可信度=可追溯性(LEDGER 红线),raw 首行固定记录 环境/被测 SHA/
# 证据仓 SHA/完整命令/UTC 时间/GPU 型号/驱动/seed——事后任何数字都能回答
# "哪台机器、哪份代码、哪条命令、哪个种子"。被 preflight.sh 与实验驱动脚本 source。
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
iso_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
utc_stamp() { date -u +%Y%m%dT%H%M%S; }
prov_line() {
  local cmd_text=${1:?usage: prov_line '<cmd>' [seed]}; local seed=${2:-none}
  # sha=被测 SGLang 上游 worktree;worktree 不在(纯 wheel 安装)时退化为固定串,不留空。
  # evidence_sha=本证据仓 HEAD(数据由哪版脚本产出);未 commit 时标 pre-commit——红旗:先 commit 再跑正式轮。
  # gpu:sort -u+paste 把同型号多卡折叠成一项(2×4090 → 一个名字),异构时并列可见。
  printf '# provenance: env=sglang-lab sha=%s evidence_sha=%s cmd="%s" date=%s gpu="%s" driver=%s seed=%s\n' \
    "$(git -C /root/repos/sglang-v0.5.18 rev-parse --short HEAD 2>/dev/null || echo wheel-v0.5.18)" \
    "$(git -C "$project_root" rev-parse --short HEAD 2>/dev/null || echo pre-commit)" \
    "$cmd_text" "$(iso_utc)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd+ -)" \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" "$seed"
}
