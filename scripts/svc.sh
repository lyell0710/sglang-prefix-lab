#!/usr/bin/env bash
# svc.sh · 本仓唯一的服务进程生命周期入口(CLAUDE.md 铁律:进程只经本脚本管理,
# 禁 killall/模糊 pkill——双卡与 venv 同 sibling 共享,模糊匹配会误杀对面进程)。
# 脱胎于 EXP-S01（独立环境与单 worker smoke）时期的 worker_ctl 脚手架;身份三形态校验与日志轮转均为事故后加固
# (EXP-S01 并发撞车、EXP-P06（路由 × 池容量）首轮 router 存活跨臂,见各记录 §7)。
# 进程生命周期:只管理本脚本写入 runtime/<name>.pid 的进程组。
# 用法: svc.sh start <name> <gpu> <port> [extra launch_server args...]
#       svc.sh stop <name> | status <name> | wait_health <name> [secs]
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${SGLLAB_VENV:-/root/venvs/sglang-lab}
RT="$project_root/runtime"; mkdir -p "$RT"

cmd=${1:?}; name=${2:?}
pidf="$RT/$name.pid"; logf="$RT/$name.log"; cmdf="$RT/$name.cmd"

# 从落盘命令反查端口:wait_health/status 不必重复传参,cmd 文件即唯一事实源。
port_of() { grep -o -- '--port [0-9]*' "$cmdf" | awk '{print $2}'; }

case "$cmd" in
  start)
    gpu=${3:?gpu}; port=${4:?port}; shift 4     # router 用 gpu 参数占位填 none
    # 双保险拒启:pid 尚活 或 端口被占 都不启动——防两个 worker 抢同一卡/端口(EXP-S01 撞车教训)。
    if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      echo "already running: $name pid=$(cat "$pidf")"; exit 1; fi
    if ss -H -ltn "sport = :$port" | grep -q .; then echo "port $port busy"; exit 1; fi
    mod=sglang.launch_server
    if [[ "$name" == router* ]]; then mod=sglang_router.launch_router; fi   # 命名约定选模块:同一套 pid/日志/停止协议管两类进程
    launch=("$VENV/bin/python" -m "$mod" --host 127.0.0.1 --port "$port" "$@")
    # 完整命令落盘:provenance 的一部分,也是 port_of 反查的依据。
    printf '%q ' "CUDA_VISIBLE_DEVICES=$gpu" "${launch[@]}" > "$cmdf"; echo >> "$cmdf"
    if [[ -s "$logf" ]]; then mv "$logf" "$logf.prev"; fi   # 证据保全:上一段日志轮转不截断——失败现场是证据(EXP-S00（bootstrap 现场审计）超时排查即靠残留日志)
    : > "$logf"
    [[ "$gpu" == none ]] && gpu=""              # router 不占卡:CUDA_VISIBLE_DEVICES 置空 = 不可见任何 GPU
    # setsid:起独立进程组——sglang 是多进程树(scheduler/detokenizer 等),
    # stop 时按 pgid 整组收割,不留孤儿占显存;HF_HUB_OFFLINE 防实验中途偷偷联网拉权重。
    CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
      setsid nohup "${launch[@]}" >> "$logf" 2>&1 < /dev/null &
    echo $! > "$pidf"
    echo "started $name pid=$(cat "$pidf") gpu=$gpu port=$port log=$logf"
    ;;
  wait_health)
    # 轮询 /health 而非 sleep 固定时长:8B 加载+CUDA graph capture 时长波动大;
    # 进程死亡立即带尾部日志退出(exit 2),与健康超时(exit 3)可区分,上层能分诊。
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
    # 身份校验(动手前最后防线):pid 文件可能过期、pid 号可能被系统复用,
    # 杀之前必须证明"这个 pid 现在仍是本仓进程"。放行三形态:
    #   ① exe 经 readlink -f 双侧解引用后 = 本 venv 的 python 真身
    #      (uv venv 的 bin/python 是符号链接,不解引用会假阴性拒停);
    #   ② cmdline 含 sglang.launch_server(worker)或 sglang_router.launch_router(router 启动形态);
    #   ③ cmdline 以 sglang::router 开头——router 用 setproctitle 改名后的运行形态。
    # 形态③是 EXP-P06 首轮事故的修复:旧校验只认②,router 改名后被 REFUSE,
    # 第一支 router 存活跨臂,cache_aware 臂实际仍在跑 round_robin(记录 §7)。
    exe=$(readlink -f "/proc/$pid/exe" || true); cl=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    venv_py=$(readlink -f "$VENV/bin/python")
    if [[ "$exe" != "$venv_py" ]] || [[ "$cl" != *sglang.launch_server* && "$cl" != *sglang_router.launch_router* && "$cl" != sglang::router* ]]; then
      echo "REFUSE: pid $pid is not our worker ($exe)"; exit 1; fi
    # 面试点:为什么杀 pgid 而不是 pid——sglang worker 是多进程树,杀父留子会
    # orphan 占显存;start 时 setsid 正是为了让整棵树共享一个可整组收割的 pgid。
    # 先 TERM 给 30s 优雅退出(flush 日志/释放显存),仍活着才升级 KILL。
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid"
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "escalating KILL to pgid $pgid"; kill -KILL -- "-$pgid" 2>/dev/null || true; }
    rm -f "$pidf"; echo "stopped $name (pgid $pgid)"
    ;;
  *) echo "unknown cmd"; exit 1;;
esac
