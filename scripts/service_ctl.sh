#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_dir="$project_root/runtime"
python_bin=${SGLLAB_PYTHON:-/root/venvs/sglang-lab/bin/python}
mkdir -p "$runtime_dir"

usage() {
  cat <<'EOF'
usage:
  service_ctl.sh start-worker <name> <gpu> <port> <model_path> <served_model_name> [context_length] [mem_fraction]
  service_ctl.sh start-router <name> <policy> <port> <worker_url>...
  service_ctl.sh status [name]
  service_ctl.sh stop <name>
  service_ctl.sh stop-all
EOF
}

pid_is_live() {
  local pid=$1
  kill -0 "$pid" 2>/dev/null
}

port_is_busy() {
  local port=$1
  ss -H -ltn "sport = :$port" | grep -q .
}

gpu_has_compute_process() {
  local gpu=$1
  local gpu_uuid
  gpu_uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$gpu_uuid"
}

start_process() {
  local name=$1
  shift
  local pid_path="$runtime_dir/$name.pid"
  local log_path="$runtime_dir/$name.log"
  local cmd_path="$runtime_dir/$name.cmd"

  if [[ -f "$pid_path" ]]; then
    local old_pid
    old_pid=$(<"$pid_path")
    if pid_is_live "$old_pid"; then
      printf 'refusing to overwrite live service %s pid=%s\n' "$name" "$old_pid" >&2
      return 1
    fi
    rm -f "$pid_path" "$cmd_path"
  fi

  : > "$log_path"
  printf '%q ' "$@" > "$cmd_path"
  printf '\n' >> "$cmd_path"
  setsid "$@" >> "$log_path" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_path"
  sleep 1
  if ! pid_is_live "$pid"; then
    printf 'service %s exited during startup; log follows\n' "$name" >&2
    tail -n 120 "$log_path" >&2 || true
    rm -f "$pid_path"
    return 1
  fi
  printf 'started name=%s pid=%s log=%s\n' "$name" "$pid" "$log_path"
}

start_worker() {
  [[ $# -ge 5 ]] || { usage >&2; return 2; }
  local name=$1 gpu=$2 port=$3 model_path=$4 served_name=$5
  local context_length=${6:-4096}
  local mem_fraction=${7:-0.70}
  if port_is_busy "$port"; then
    printf 'refusing to start %s: port %s is already listening\n' "$name" "$port" >&2
    return 1
  fi
  if gpu_has_compute_process "$gpu"; then
    printf 'refusing to start %s: GPU %s already has a compute process\n' "$name" "$gpu" >&2
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader >&2 || true
    return 1
  fi
  start_process "$name" env \
    "CUDA_VISIBLE_DEVICES=$gpu" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "$python_bin" -m sglang.launch_server \
    --model-path "$model_path" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --tp-size 1 \
    --context-length "$context_length" \
    --mem-fraction-static "$mem_fraction" \
    --enable-metrics \
    --log-level info
}

start_router() {
  [[ $# -ge 4 ]] || { usage >&2; return 2; }
  local name=$1 policy=$2 port=$3
  shift 3
  if port_is_busy "$port"; then
    printf 'refusing to start %s: port %s is already listening\n' "$name" "$port" >&2
    return 1
  fi
  start_process "$name" "$python_bin" -m sglang_router.launch_router \
    --worker-urls "$@" \
    --policy "$policy" \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.5 \
    --eviction-interval-secs 120 \
    --max-tree-size 67108864 \
    --host 127.0.0.1 \
    --port "$port" \
    --prometheus-host 127.0.0.1 \
    --log-level info
}

status_one() {
  local name=$1
  local pid_path="$runtime_dir/$name.pid"
  if [[ ! -f "$pid_path" ]]; then
    printf '%s stopped (no pid file)\n' "$name"
    return 1
  fi
  local pid
  pid=$(<"$pid_path")
  if pid_is_live "$pid"; then
    ps -o pid=,ppid=,sid=,stat=,etime=,args= -p "$pid"
  else
    printf '%s stale pid=%s\n' "$name" "$pid"
    return 1
  fi
}

stop_one() {
  local name=$1
  local pid_path="$runtime_dir/$name.pid"
  [[ -f "$pid_path" ]] || { printf '%s already stopped\n' "$name"; return 0; }
  local pid
  pid=$(<"$pid_path")
  if ! pid_is_live "$pid"; then
    printf '%s had stale pid=%s\n' "$name" "$pid"
    rm -f "$pid_path" "$runtime_dir/$name.cmd"
    return 0
  fi

  local cmdline
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  if [[ "$cmdline" != *sglang* ]]; then
    printf 'refusing to signal pid=%s: command is not an SGLang service: %s\n' "$pid" "$cmdline" >&2
    return 1
  fi

  kill -TERM -- "-$pid"
  for _ in $(seq 1 30); do
    pid_is_live "$pid" || break
    sleep 1
  done
  if pid_is_live "$pid"; then
    printf 'service %s did not exit after 30s; leaving it for manual inspection\n' "$name" >&2
    return 1
  fi
  rm -f "$pid_path" "$runtime_dir/$name.cmd"
  printf 'stopped name=%s pid=%s\n' "$name" "$pid"
}

case ${1:-} in
  start-worker)
    shift; start_worker "$@" ;;
  start-router)
    shift; start_router "$@" ;;
  status)
    shift
    if [[ $# -eq 1 ]]; then status_one "$1"; else
      found=0
      for pid_path in "$runtime_dir"/*.pid; do
        [[ -e "$pid_path" ]] || continue
        found=1
        status_one "$(basename "$pid_path" .pid)" || true
      done
      [[ $found -eq 1 ]] || printf 'no managed services\n'
    fi ;;
  stop)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    stop_one "$2" ;;
  stop-all)
    for pid_path in "$runtime_dir"/*.pid; do
      [[ -e "$pid_path" ]] || continue
      stop_one "$(basename "$pid_path" .pid)"
    done ;;
  *)
    usage >&2; exit 2 ;;
esac
