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

## 2026-08-24 · EXP-S01 独立环境与单 worker smoke

- **做了什么**：建立 `/root/venvs/sglang-lab`，锁 SGLang v0.5.18/router 0.3.2；创建 detached upstream
  worktree；实现精确 PID/进程组生命周期、health wait、smoke/metrics/raw 捕获并完成 GPU1 隔离复跑。
- **为什么**：先以 0.6B 打通 CUDA/JIT/Graph/API/metrics，再让 8B 承担正式矩阵，降低环境排障成本。
- **关键数字**：204 packages 依赖检查通过；worker 稳态 9,914 MiB，KV pool 64,403 tokens；API/metrics PASS。
  均为 smoke 事实，不能当性能数字。
- **异常**：第一次运行与自动恢复的另一 Claude agent 同时在 GPU0 启动 worker，双方 OOM；两侧日志保留，
  correction raw 明确判为协议碰撞。隔离到 GPU1 后默认 graph 配置 PASS。
- **产物路径**：records/EXP-S01、`data/raw/EXP-S01/`、scripts/{capture_env,service_ctl,wait_http,
  smoke_client,run_smoke,archive_runtime_log}。
- **下一步**：EXP-S02 固化 GSP/agentic workload manifest、token 长度和逐 token reference。
