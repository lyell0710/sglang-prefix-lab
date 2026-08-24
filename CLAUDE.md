@/root/standards/CORE.md
动手实验前完整阅读 `/root/standards/STANDARDS.md` 的实验/benchmark/图表/原理笔记章节。

本项目附则:
- 被测基线固定 SGLang v0.5.18(venv `/root/venvs/sglang-lab`);未经单独 EXP 不升级。
- **本机存在并行的 sibling agent(Codex)实验仓 `/root/projects/sglang-inference-lab`**,
  共用同一对 4090 与同一 venv。本仓端口错开(worker 28000/28001、router 40000),
  headline 双副本实验需要独占整机——运行前 preflight 必须确认两卡空闲且无他人 sglang 进程。
- 进程只经 `scripts/svc.sh` 管理(写 runtime/<name>.pid);禁止 killall/模糊 pkill;
  停止前用 /proc/<pid>/exe 校验是本仓 venv 的 launch_server 才动手。
- 每个性能点 ≥3 独立 round;profile run 永不进性能汇总;raw 不覆盖。
