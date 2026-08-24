#!/usr/bin/env bash
# LEGACY bootstrap helper retained as EXP-S01 incident evidence.
# 进程生命周期:只管理本脚本写入 runtime/<name>.pid 的进程组。
# 用法: worker_ctl.sh start <name> <gpu> <port> [extra launch_server args...]
#       worker_ctl.sh stop <name> | status <name> | wait_health <name> [secs]
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${SGLLAB_VENV:-/root/venvs/sglang-lab}
RT="$project_root/runtime"; mkdir -p "$RT"

cmd=${1:?}; name=${2:?}
pidf="$RT/$name.pid"; logf="$RT/$name.log"; cmdf="$RT/$name.cmd"

port_of() { grep -o -- '--port [0-9]*' "$cmdf" | awk '{print $2}'; }

case "$cmd" in
  start)
    gpu=${3:?gpu}; port=${4:?port}; shift 4     # router 用 gpu 参数占位填 none
    if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      echo "already running: $name pid=$(cat "$pidf")"; exit 1; fi
    if ss -H -ltn "sport = :$port" | grep -q .; then echo "port $port busy"; exit 1; fi
    mod=sglang.launch_server
    if [[ "$name" == router* ]]; then mod=sglang_router.launch_router; fi
    launch=("$VENV/bin/python" -m "$mod" --host 127.0.0.1 --port "$port" "$@")
    printf '%q ' "CUDA_VISIBLE_DEVICES=$gpu" "${launch[@]}" > "$cmdf"; echo >> "$cmdf"
    if [[ -s "$logf" ]]; then mv "$logf" "$logf.prev"; fi   # 证据保全:上一段日志轮转不截断
    : > "$logf"
    [[ "$gpu" == none ]] && gpu=""
    CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
      setsid nohup "${launch[@]}" >> "$logf" 2>&1 < /dev/null &
    echo $! > "$pidf"
    echo "started $name pid=$(cat "$pidf") gpu=$gpu port=$port log=$logf"
    ;;
  wait_health)
    secs=${3:-600}; port=$(port_of)
    for ((i=0; i<secs; i+=5)); do
      if curl -sf -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "healthy after ${i}s"; exit 0; fi
      if ! kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "process died; see $logf"; tail -20 "$logf"; exit 2; fi
      sleep 5
    done
    echo "health timeout after ${secs}s"; exit 3
    ;;
  status)
    if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      pid=$(cat "$pidf"); echo "running pid=$pid pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')"
      ps -o pid=,stat=,etime=,args= --ppid "$pid" | cut -c1-120
    else echo "not running"; fi
    ;;
  stop)
    [[ -f "$pidf" ]] || { echo "no pid file"; exit 0; }
    pid=$(cat "$pidf")
    if ! kill -0 "$pid" 2>/dev/null; then echo "not running"; rm -f "$pidf"; exit 0; fi
    # 身份校验:必须是本 venv 的 python 且在跑 sglang.launch_server
    exe=$(readlink -f "/proc/$pid/exe" || true); cl=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    venv_py=$(readlink -f "$VENV/bin/python")   # uv venv 的 python 是符号链接,比真身
    # 注意:sglang-router 会 setproctitle 成 "sglang::router",cmdline 不含模块名
    if [[ "$exe" != "$venv_py" ]] || [[ "$cl" != *sglang.launch_server* && "$cl" != *sglang_router.launch_router* && "$cl" != sglang::router* ]]; then
      echo "REFUSE: pid $pid is not our worker ($exe)"; exit 1; fi
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid"
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "escalating KILL to pgid $pgid"; kill -KILL -- "-$pgid" 2>/dev/null || true; }
    rm -f "$pidf"; echo "stopped $name (pgid $pgid)"
    ;;
  *) echo "unknown cmd"; exit 1;;
esac
