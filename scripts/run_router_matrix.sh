#!/bin/bash
# EXP-S04 主矩阵 driver：2 policy × 3 workload × 3 concurrency × 3 seeds = 54 cell。
# 每 cell 跑 scripts/bench_router_matrix.py 落盘 data/raw/EXP-S04/。
# 里程碑即提交：全部跑完汇总 + 提交。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/root/venvs/sglang-lab/bin/python
MD=data/raw/EXP-S02
OUT=data/raw/EXP-S04
mkdir -p "$OUT"

for policy in round_robin cache_aware; do
  # 停上一轮 router（若存在），启动本 policy router
  for rn in router_rr router_ca; do
    [[ -f runtime/$rn.pid ]] && bash scripts/svc.sh stop "$rn" 2>/dev/null || true
  done
  rn="router_${policy/rr/rr}"   # round_robin→router_rr, cache_aware→router_ca
  rn=$(echo "$policy" | sed 's/round_robin/router_rr/; s/cache_aware/router_ca/')
  bash scripts/svc.sh start "$rn" none 40000 \
    --worker-urls http://127.0.0.1:28000 http://127.0.0.1:28001 \
    --policy "$policy" 2>&1 | tail -1
  sleep 8
  for workload in unique_control hot_prefix_1024 hot_prefix_1792; do
    for c in 1 4 16; do
      for seed in 2026082401 2026082402 2026082403; do
        echo "=== $policy $workload c$c s$seed ==="
        $PY scripts/bench_router_matrix.py --policy "$policy" --workload "$workload" \
          --concurrency "$c" --seed "$seed" --manifest-dir "$MD" \
          --base-url http://127.0.0.1:40000 --model Qwen/Qwen3-8B --out "$OUT" \
          2>&1 | tail -3
      done
    done
  done
done
echo "MATRIX_DONE"
