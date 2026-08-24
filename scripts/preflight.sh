#!/usr/bin/env bash
# 无副作用体检;必须在任何 GPU 实验前跑。发现外来 GPU 进程/端口占用 → 非零退出。
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/provenance.sh"
out=${1:-}
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
fail=0
{
  prov_line "bash scripts/preflight.sh${out:+ $out}"
  printf 'date_utc=%s host=%s kernel=%s\n' "$(iso_utc)" "$(hostname)" "$(uname -r)"
  printf '\n[gpu]\n'; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate,temperature.gpu,power.limit --format=csv,noheader
  printf '\n[gpu_compute_processes]\n'; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
  printf '\n[foreign_gpu_procs]\n'
  (set +e   # /proc 竞态:进程随时消失,本段禁 set -e 连坐
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    if ls -la "$p/fd" 2>/dev/null | grep -q nvidia; then
      exe=$(readlink "$p/exe" 2>/dev/null); cl=$(tr "\0" " " < "$p/cmdline" 2>/dev/null)
      printf '%s %s :: %.120s\n' "$pid" "$exe" "$cl"
    fi
  done; true)
  printf '\n[ports]\n'
  for port in 28000 28001 40000 29000; do
    if ss -H -ltn "sport = :$port" | grep -q .; then printf 'port_%s=busy\n' "$port"; else printf 'port_%s=free\n' "$port"; fi
  done
  printf '\n[sibling_lab]\n'
  printf 'sibling_head=%s\n' "$(git -C /root/projects/sglang-inference-lab rev-parse --short HEAD 2>/dev/null || echo n/a)"
  ps -e -o pid=,args= | awk '/sglang.launch_server|sglang_router|sgl-model-gateway/ && $0 !~ /awk/ {print}' || true
  printf '\n[upstream]\n'
  git -C /root/repos/sglang-v0.5.18 rev-parse --short HEAD 2>/dev/null || true
} > "$tmp"
# 判定:GPU 显存被外来进程占用 / 本仓端口 busy 视为不通过
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "PRECHECK_FAIL: GPU busy (compute processes present)" >> "$tmp"; fail=1; fi
if grep -q 'port_.*=busy' "$tmp"; then echo "PRECHECK_FAIL: project port busy" >> "$tmp"; fail=1; fi
[[ -n "$out" ]] && { mkdir -p "$(dirname "$out")"; cp "$tmp" "$out"; }
cat "$tmp"
exit $fail
