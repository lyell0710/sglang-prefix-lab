# 工程决策记录

## D-001 · 稳定 tag 作为性能基线

- 日期：2026-08-24
- 决策：实验锁定 SGLang v0.5.18，不直接使用当天 main。
- 原因：tag 可复现且有预编译 CUDA 13 wheel；main 变化快并需要额外 Rust source-build 条件。
- 后果：upstream main 只用于读源码/查重；任何 main 对照必须单列 EXP，不能混入 v0.5.18 汇总。

## D-002 · 证据仓与 upstream 分离

- 日期：2026-08-24
- 决策：证据仓位于 `/root/projects/sglang-inference-lab`；upstream checkout 位于 `/root/repos/`。
- 原因：避免日志/raw 污染上游分支，也与 vLLM staged 配置完全隔离。
- 后果：provenance 同时记录被测 SGLang SHA 和证据仓 SHA；上游修改只进入专用 worktree/branch。

## D-003 · 不接管来源不明的进程

- 日期：2026-08-24
- 决策：服务脚本只管理自身写入 `runtime/` 的 PID；不使用 killall/模糊 pkill。
- 原因：共享机器上有 Jupyter、Cursor/Claude 和其他项目守护进程。

