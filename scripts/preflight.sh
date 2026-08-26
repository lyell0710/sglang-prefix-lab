#!/usr/bin/env bash
# 无副作用体检:任何 GPU 实验前必须跑(CLAUDE.md 铁律)。零写操作,只读取证;
# 发现外来 GPU compute 进程或本仓端口被占 → 非零退出,上层流程据此中止。
# 为什么必须:双卡与 venv 同 sibling 仓共享,"整机独占"只能靠这里把关——
# EXP-S01（独立环境与单 worker smoke）的并发撞车事故(另一 agent 起了第二个 worker 抢同一卡)即反面教材。
# 输出即证据:可传 $1 落盘为 raw preflight 快照(provenance 首行),与实验数据同谱系归档。
# 面试点:判定与取证分离——判定用 nvidia-smi 计数(稳定),取证用 /proc 扫描
# (能点名 exe/cmdline,但有进程消失竞态,故 set +e 防连坐)。
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
  # 取证段:nvidia-smi 只给 pid;扫 /proc 的 fd 中 nvidia 句柄可点名"谁在用卡"
  # (exe+cmdline),事故时能直接指认来源进程。
  (set +e   # /proc 竞态:进程随时消失,本段禁 set -e 连坐
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    if ls -la "$p/fd" 2>/dev/null | grep -q nvidia; then
      exe=$(readlink "$p/exe" 2>/dev/null); cl=$(tr "\0" " " < "$p/cmdline" 2>/dev/null)
      printf '%s %s :: %.120s\n' "$pid" "$exe" "$cl"
    fi
  done; true)
  printf '\n[ports]\n'
  # 本仓声明端口(28000/28001=w0/w1,40000=router,29000=预留);busy 即上一轮未清理或外来占用。
  for port in 28000 28001 40000 29000; do
    if ss -H -ltn "sport = :$port" | grep -q .; then printf 'port_%s=busy\n' "$port"; else printf 'port_%s=free\n' "$port"; fi
  done
  printf '\n[sibling_lab]\n'
  # 记录 sibling 仓 HEAD 与任何存活的 sglang 进程:共享硬件,证据里必须能看到"当时对面在干什么"。
  printf 'sibling_head=%s\n' "$(git -C /root/projects/sglang-inference-lab rev-parse --short HEAD 2>/dev/null || echo n/a)"
  ps -e -o pid=,args= | awk '/sglang.launch_server|sglang_router|sgl-model-gateway/ && $0 !~ /awk/ {print}' || true
  printf '\n[upstream]\n'
  # 被测 SGLang 源码 SHA 入证据:防"测的是哪个版本"失溯。
  git -C /root/repos/sglang-v0.5.18 rev-parse --short HEAD 2>/dev/null || true
} > "$tmp"
# 判定:GPU 显存被外来进程占用 / 本仓端口 busy 视为不通过。FAIL 行写进快照本身
# (而非只留退出码):落盘证据自含结论,事后无需复原当时上下文。
# 不区分自家/外来 compute 进程:实验要求整机独占,任何进程在卡上都不许开跑。
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "PRECHECK_FAIL: GPU busy (compute processes present)" >> "$tmp"; fail=1; fi
if grep -q 'port_.*=busy' "$tmp"; then echo "PRECHECK_FAIL: project port busy" >> "$tmp"; fail=1; fi
[[ -n "$out" ]] && { mkdir -p "$(dirname "$out")"; cp "$tmp" "$out"; }
cat "$tmp"
exit $fail
