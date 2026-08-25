@/root/standards/CORE.md

动手实验前完整阅读 `/root/standards/STANDARDS.md` 的实验、benchmark、图表和原理笔记章节。

本项目附则：

- 被测基线固定 SGLang v0.5.18；未经单独 EXP 不升级。
- 禁止 `killall_sglang` 或模糊 `pkill`；只停止 `runtime/` 中由本项目记录的 PID。
- benchmark、server log、router metrics、GPU telemetry 共用同一秒级 run prefix。
- 每个性能点至少 3 个独立 round；profile run 永不进入性能汇总。
- 修改 upstream 只能在专用 branch/worktree；本证据仓不得混入 upstream 源码副本。

