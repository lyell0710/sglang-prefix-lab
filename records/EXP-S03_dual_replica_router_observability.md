# EXP-S03 · dual-replica router observability（双 worker + router 可观测性）

> **一句话结论**：双 worker（8B × 2）+ router 的可观测性全链路打通——`/workers`、`/get_loads`、`/metrics` 均可采，round_robin 与 cache_aware 两种策略的 parity probe 均逐 token 一致（关 thinking 后），S04 主矩阵的地基就绪。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | venv sglang-lab · GPU0/1 · Qwen3-8B(bf16, revision b968826…) @ 28000/28001 · mem-fraction 0.85 · router @ 40000 |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN_router_matrix.md EXP-S03 |

## 1. 目的与假设

双 worker + router 启动/清理通过；`/workers`、`/get_loads`、Prometheus metrics 可采；两种策略都过 parity probe。PASS 判定：两种策略下 probe 输出与单 worker reference 逐 token 一致。

## 2. 环境与配置

- worker：`svc.sh start w{0,1} {0,1} 2800{0,1} --model-path …/Qwen3-8B/… --served-model-name Qwen/Qwen3-8B --mem-fraction-static 0.85 --context-length 4096 --tp-size 1 --enable-metrics`。
- router：`svc.sh start router_{rr,ca} none 40000 --worker-urls http://127.0.0.1:28000 http://127.0.0.1:28001 --policy {round_robin,cache_aware}`。

## 3. 步骤

1. 启动双 worker（加 `--enable-metrics`）。
2. 启动 router（round_robin），验 `/workers`/`/get_loads`/`/metrics`。
3. parity probe（round_robin）→ 切 cache_aware → parity probe。

## 4. 原始数据

- `runtime/{w0,w1,router_rr,router_ca}.log`（启动日志，含 policy 与 workers 注册）。
- parity probe 输出（终端级，与 reference.json 逐 token 比对）。

## 5. 结果

| 检查 | 结果 |
|---|---|
| `/workers` | 200，2 worker 均 `is_healthy=true`、`load=0`、`load_balance_method=round_robin`（worker 侧） |
| `/get_loads` | 200，`{"worker":"…28000","load":0}` × 2 |
| `/metrics` | 200（`--enable-metrics` 后），`sglang:prompt_tokens_total` 可采 |
| router policy 标签 | round_robin 臂与 cache_aware 臂均从 router 日志确认（`policy: CacheAware { cache_threshold: 0.3, … }`） |
| parity（round_robin） | probe 输出与 reference 逐 token 一致（关 thinking 后） |
| parity（cache_aware） | 同上，一致 |

## 6. 分析与结论

- **可观测性链路完整**：router 的 `/workers` 给出 worker 健康态 + 元数据（含 load_balance_method），`/get_loads` 给出实时负载，worker `/metrics` 给 `prompt_tokens_total`（用于流量落点差分，bench_route_pool 同法）。三条正是 S04 归因「路由决策落点」的证据来源。
- **parity 通过的前提是关 thinking**：Qwen3 的 enable_thinking 非确定（EXP-S02 §7 勘误），关掉后两种策略的 probe 都与 reference 逐 token 一致——router 不改变 worker 的确定性行为。

## 7. 异常、偏差与开放问题

- **worker metrics 默认关**：v0.5.18 `enable_metrics=False`，不加 `--enable-metrics` 则 /metrics 404。S04 的 bench 若需 metrics 差分，worker 必须带此参数。
- worker metadata 的 `load_balance_method` 恒为 `round_robin`（worker 侧默认），不反映 router policy——router policy 只能从 router 日志 `policy: …` 行确认（bench_route_pool 的 policy 标签 gate 同源）。
- cache_aware 的平衡阈值（`balance_abs_threshold: 64`）是 sglang 默认，S04 若发现失衡回退需在记录标注。

## 8. 下游影响

- 解锁 S04 主矩阵的「同一 manifest」「parity probe」「policy 标签」三条 gate。
- S04 直接复用本 worker/router 拓扑（28000/28001/40000），bench 客户端从 manifest 读请求、经 router 发、逐请求记 cached_tokens + TTFT/TPOT/E2E。
