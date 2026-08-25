#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/provenance.sh"

output_path=${1:-}
tmp_path=$(mktemp)
trap 'rm -f "$tmp_path"' EXIT

{
  prov_line "bash scripts/preflight.sh${output_path:+ $output_path}"
  printf 'project_root=%s\n' "$project_root"
  printf 'date_utc=%s\n' "$(iso_utc)"
  printf 'host=%s\n' "$(hostname)"
  printf 'kernel=%s\n' "$(uname -r)"
  printf 'python=%s\n' "$(python3 --version 2>&1)"
  printf 'uv=%s\n' "$(uv --version 2>&1 || true)"
  printf 'cuda=%s\n' "$(nvcc --version 2>/dev/null | sed -n 's/.*release \([^,]*\).*/\1/p')"
  printf 'disk_root='; df -h / | awk 'NR==2 {print $3 "/" $2 " used=" $5 " available=" $4}'
  printf '\n[gpu]\n'
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,pstate,temperature.gpu,power.limit \
    --format=csv,noheader
  printf '\n[gpu_compute_processes]\n'
  nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || true
  printf '\n[ports]\n'
  for port in 18000 18001 30000; do
    if ss -H -ltn "sport = :$port" | grep -q .; then
      printf 'port_%s=busy\n' "$port"
    else
      printf 'port_%s=free\n' "$port"
    fi
  done
  printf '\n[uninterruptible_tasks]\n'
  ps -e -o pid=,ppid=,stat=,wchan:28=,comm= | awk '$3 ~ /^D/ {print}' || true
  printf '\n[sglang_processes]\n'
  ps -e -o pid=,ppid=,stat=,args= | awk '/sglang|sgl_router/ && $0 !~ /awk/ {print}' || true
  printf '\n[upstream]\n'
  for upstream_dir in /root/repos/sglang-v0.5.18 /root/repos/sglang; do
    if git -C "$upstream_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      printf '%s sha=%s describe=%s dirty=%s\n' "$upstream_dir" \
        "$(git -C "$upstream_dir" rev-parse HEAD)" \
        "$(git -C "$upstream_dir" describe --always --tags 2>/dev/null)" \
        "$(test -z "$(git -C "$upstream_dir" status --porcelain)" && printf no || printf yes)"
    fi
  done
} > "$tmp_path"

if [[ -n "$output_path" ]]; then
  mkdir -p "$(dirname "$output_path")"
  cp "$tmp_path" "$output_path"
fi

sed -n '1,240p' "$tmp_path"

