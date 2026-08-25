@/root/standards/CORE.md
动手实验前完整阅读 `/root/standards/STANDARDS.md` 的实验/benchmark/图表/原理笔记章节。

本项目附则:
- 被测基线固定 SGLang v0.5.18(venv `/root/venvs/sglang-lab`);未经单独 EXP 不升级。
- sibling 仓已并入本仓(2026-08-25,router-lab 合并后拍平);旧路径只剩指路牌。
  若 Codex agent 恢复活动会看到指路牌——仍保持:任何 GPU 实验前 preflight 必须
  确认两卡空闲且无外来 sglang 进程;本仓端口 28000/28001/40000/29000。
- 进程只经 `scripts/svc.sh` 管理(写 runtime/<name>.pid);禁止 killall/模糊 pkill;
  停止前用 /proc/<pid>/exe 校验是本仓 venv 的 launch_server 才动手。
- 每个性能点 ≥3 独立 round;profile run 永不进性能汇总;raw 不覆盖。
- README=对外门面(面试官视角);状态/红线唯一权威=LEDGER.md。
