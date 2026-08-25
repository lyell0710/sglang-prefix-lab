# Agent instructions

先读 `/root/standards/CORE.md` 与 `/root/standards/STANDARDS.md`，再读本仓 `CLAUDE.md`、
`README.md` 和 `docs/PLAN.md`。

- README 是状态唯一来源；不要在多份文档维护两套数字。
- 不接管来源不明的 GPU/服务进程。启动前运行 `scripts/preflight.sh`。
- 不修改 `/root/projects/vllm` 及其 staged 文件。
- raw 不覆盖；任何失败和协议偏离都进入对应 EXP 记录。

