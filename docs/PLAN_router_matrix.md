> 迁移注(2026-08-25):本协议为并入的 router 性能矩阵预注册计划(S02-S07,未执行),原仓 sglang-inference-lab;机读配置=config/protocol-router-v1.json;执行前须按本仓 CORE 附则复核端口与资源纪律。

# 实验计划 · Prefix locality × routing policy × load

> 协议版本 v1，2026-08-24 预注册。若后续修改参数，必须新建协议版本并在 EXP 记录中解释，不能覆盖本文件历史。

## 研究问题

在两个同构 SGLang worker 上，前缀感知路由是否能把同一热前缀稳定送到已有 KV 的 worker，从而降低 TTFT？
当并发升高或热门前缀偏斜时，cache reuse 与 load balance 的交界在哪里？

## 固定变量

- SGLang v0.5.18；Qwen3-0.6B 只做 smoke，正式矩阵使用固定 revision 的 Qwen3-8B；BF16/greedy。
- GPU0/GPU1 各一个 TP=1 worker；相同 server 参数和显存配额。
- CUDA Graph 先使用 v0.5.18 默认；每个模型的 `mem-fraction-static` 只在单进程 smoke 后锁定，所有路由臂保持一致。
- 每次 A/B 使用同一请求 manifest、相同 seed、相同 worker 冷启动和固定 warmup。
- client 与两 worker 同机，因此只比较策略，不外推网络集群。

机器可读配置见 `config/protocol-v1.json`。

## 主矩阵

| 维度 | 水平 |
|---|---|
| 路由策略 | `round_robin`, `cache_aware` |
| 前缀局部性 | 总长均为 2 Ki-token：unique/control、1 Ki-token hot prefix、1.75 Ki-token hot prefix |
| client concurrency | 1、4、16 |
| 独立 round | 3 个配对 seed；每个 seed 内策略顺序随机化 |

每个正式点 192 requests，满足官方 `num_prompts >= 5 × max_concurrency` 建议。输出长度固定 32 token；
具体 tokenizer 后的长度由 EXP-S02 manifest 验证，超出上下文的请求在计时前即判协议 FAIL。

## Gate（跑前锁定）

### 正确性与完整性

- expected/completed request 数完全相等，HTTP/engine 失败为 0。
- 固定 probe 集在 temperature=0 下与单 worker reference 逐 token 一致。
- 请求 manifest SHA256 在 A/B 两臂相同；模型 revision、server args 和 seed 相同。
- 目标 policy 由 router 启动日志/配置快照证明；silent fallback 必须为 false。

### 测量有效性

- JIT/模型 warmup 在正式计时外完成；每个 arm 从等价进程生命周期开始。
- profiler 关闭；profile run 写 `PROFILE_ONLY`，不进 derived。
- benchmark 前后捕获 router/worker metrics 与 GPU telemetry；GPU 上无项目外 compute process。
- A/B 使用同一 power limit；配对运行的温度差 ≤5°C，median SM clock 差 ≤5%，否则该 pair 失败并保留 raw。
- 每点至少 3 个独立 round；报告 mean±std，同时保留 p50/p95/p99，不只报最佳值。

### 主张阈值

- 相对变化绝对值 <5% 或与跨 round 波动同量级时，结论写“无可区分改善”，不写提速。
- shared-prefix 正向结果必须伴随 unique-prefix 反例；否则不能归因于 prefix reuse。
- 只要正确性、完整性或 codepath gate 失败，该 run 永不进入图、README headline 或简历。

## EXP 清单

### EXP-S00 · bootstrap audit

固化 host/GPU/端口/进程、磁盘、网络和半成品状态；解释截图超时是否来自 GPU、网络或 I/O。

### EXP-S01 · environment and single-worker smoke

建立独立 venv，冻结包版本；单 worker 依次通过 `/health`、`/v1/models`、确定性 completion；记录启动日志和 GPU 显存。

### EXP-S02 · correctness and workload contract

生成不可变 JSONL manifest；验证 tokenized prefix/suffix/total 长度；采单 worker reference token IDs 与响应 digest。

### EXP-S03 · dual-replica router observability

双 worker + router 启动/清理脚本通过；`/workers`、`/get_loads`、Prometheus metrics 可采；两种策略都过 parity probe。

### EXP-S04 · routing policy matrix

完成主矩阵。直接测量 TTFT/TPOT/E2E/吞吐/失败率，以及每 worker request/load/cache 指标；只由 gate PASS raw 生成汇总。

### EXP-S05 · boundary and profile attribution

围绕 S04 的反转点做 Zipf/阈值最小附加 sweep；用独立 profile run 验证是 prefix prefill、排队还是路由开销主导，
不混用 profile 时延。最后用官方 agentic-trace 或固定 multi-turn replay 检查受控 GSP 结论能否外推到会话型流量。

### EXP-S06 · repeatability and resume evidence

全流程从 manifest 重放；重算图表；形成一句带硬件、模型、工作负载和证据边界的简历草案。

### EXP-S07 · upstream gap and PR gate

仅在实验暴露可复现工程缺口时开展：查 issue/PR、写最小测试、feature branch 验证。没有真实缺口时允许诚实结束，不为 PR 而造 PR。
