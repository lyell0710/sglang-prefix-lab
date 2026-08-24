# LAB JOURNAL · SGLang Inference Lab

每个工作段落追加：做了什么 / 为什么 / 关键数字 / 产物路径 / 下一步。时间正序，不回写历史。

---

## 2026-08-24 · bootstrap 与现场体检

- **做了什么**：核对已有 SGLang clone、所有 venv、GPU/端口/进程、模型缓存、磁盘与网络；读取统一工程规范；
  预注册“共享前缀 × 路由策略 × 负载”的项目主线并建立独立证据仓。
- **为什么**：截图中的 subprocess 初始化在 60 秒超时，必须先排除正在运行的 kernel、端口冲突和半安装环境，
  同时避免污染现有 vLLM 项目。
- **关键数字**：两张 RTX 4090 均 0% 利用率且无 compute process；网络探测可达；现场高 I/O wait 与
  一个遗留全盘 `rg` 及两条认证检查阻塞同时出现。事故高峰为终端级证据；清理后状态已落 EXP-S00 raw。
- **产物路径**：本仓骨架、`docs/PLAN.md`、`config/protocol-v1.json`、records/EXP-S00 及配对 raw。
- **下一步**：捕获 EXP-S00 raw；等待 I/O 恢复后创建独立 venv，安装固定 v0.5.18 并完成单 worker smoke。
