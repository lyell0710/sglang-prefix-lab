# EXP-S04 · routing policy matrix（主矩阵 54 cell）

> **一句话结论**：主矩阵 54 cell 全跑完，但暴露一个**决定性的测量缺陷**：v0.5.18 经 router 的响应 `prompt_tokens_details=null`（cached_tokens 被 router 重序列化丢弃），cache_aware 是否真的命中前缀缓存**无法从响应判定**——本批 TTFT 数据只能算「策略选择下的时延」，不能证明「cache_aware 命中缓存」。且 cache_aware 在高并发下 TTFT 反而 +53%~+196%，与「命中降低 TTFT」的预期相反，需 worker 侧 metrics 差分才能归因。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | venv sglang-lab · GPU0/1 · Qwen3-8B @ 28000/28001 · router @ 40000 |
| 状态 | **部分完成（数据落盘，cache 命中证据缺失，待补）** |
| 关联清单项 | docs/PLAN_router_matrix.md EXP-S04 |

## 1. 目的与假设

跑主矩阵（2 policy × 3 workload × 3 concurrency × 3 seeds = 54 cell），测 TTFT/TPOT/E2E/吞吐。预期（预注册）：cache_aware 在 hot_prefix 工作负载下把同一前缀钉到一张卡 → 命中缓存 → TTFT 下降；unique_control 无前缀可共享 → 两策略持平。

## 2. 环境与配置

- worker 双卡 8B，`--enable-metrics`；router 按 policy 切换（round_robin / cache_aware）。
- bench：`scripts/bench_router_matrix.py`（stream 逐 token 计时 TTFT/TPOT/E2E，192 请求/点，温度 0，关 thinking）。

## 3. 步骤

`scripts/run_router_matrix.sh`（后台跑，约 64 分钟）→ `scripts/aggregate_router_matrix.py` 聚合。

## 4. 原始数据

- `data/raw/EXP-S04/{policy}_{workload}_c{concurrency}_s{seed}.json`（54 份，每份 192 请求逐条）。
- `data/raw/EXP-S04/derived_matrix.csv`（18 cell × 3 seeds 聚合）。

## 5. 结果

TTFT p50（ms），3 seeds 聚合：

| workload | c | round_robin | cache_aware | Δ% |
|---|---|---|---|---|
| unique_control | 1 | 298.9 | 302.6 | +1.2 |
| unique_control | 4 | 425.6 | 747.6 | **+75.7** |
| unique_control | 16 | 754.3 | 2231.5 | **+195.8** |
| hot_prefix_1024 | 1 | 198.5 | 202.3 | +1.9 |
| hot_prefix_1024 | 4 | 230.6 | 451.3 | **+95.7** |
| hot_prefix_1024 | 16 | 725.3 | 1111.6 | +53.3 |
| hot_prefix_1792 | 1 | 109.9 | 112.2 | +2.1 |
| hot_prefix_1792 | 4 | 130.7 | 221.9 | **+69.8** |
| hot_prefix_1792 | 16 | 275.1 | 413.6 | +50.3 |

（TPOT/E2E 同向；c1 两策略基本持平，c≥4 cache_aware 全面变差。）

## 6. 分析与结论

**① 决定性缺陷：cached_tokens 被 router 丢弃，cache 命中无法从响应验证。** 经 router 的响应 `prompt_tokens_details=null`（直接打 worker 有值），router 按 OpenAI schema 重序列化时丢了这个字段——EXP-P06 的「input_ids 被丢弃」同型问题的变体，只是这次丢的是**响应侧**字段而非请求侧。后果：本批 54 cell 的 TTFT 只能证明「在某策略下跑」，不能证明「cache_aware 命中/没命中」。

**② cache_aware 高并发 TTFT 变差的反常，候选解释（均未定案，需 worker metrics 差分）：**
- **负载不均衡**：cache_aware 把前缀钉到一张卡，高并发下该卡过载排队，另一卡空闲——`balance_abs_threshold=64` 的失衡回退在热前缀高度偏斜时可能未触发或触发不足。这解释「hot_prefix 且 c≥4 变差」的模式（前缀集中 → 单卡过载）。
- **基准不公平**：round_robin 在 hot_prefix 下反而是「完美分片」（EXP-P06 的奇偶巧合），自然均衡；cache_aware 的「均衡」依赖回退机制，c≥4 时可能失效。
- 但**无法排除测量本身**：stream 并发下 `sem` 限流 + 192 请求的排队在两种 policy 下可能不同。

**③ 结论暂缓。** 在补上 cached_tokens（或 worker metrics 差分）之前，本矩阵**不能下任何「cache_aware vs round_robin」的性能结论**。54 cell 数据本身有效（时延可复现），但归因链缺了「命中」这一环。

## 7. 异常、偏差与开放问题

- **核心开放问题**：v0.5.18 router 响应丢 `prompt_tokens_details`。补证三选一：① 直接打 worker 的 cached_tokens（绕过 router，但那样就不是「经 router 的 cache_aware」）；② worker 侧 `/metrics` 的 `prompt_tokens_total` 差分（bench_route_pool 同法，能反推流量落点但不能证明命中）；③ 检查 sglang 是否有响应透传开关。
- 吞吐字段 `throughput_req_s` 在 c=1 时的公式有误（用了 e2e 累加而非并发归一），需订正，但不影响 TTFT/TPOT 主结论。
- stream 模式拿不到 cached_tokens 是设计缺陷，S05 若要命中证据需改用非 stream + 直接 worker 或 metrics 差分。

## 8. 下游影响

- S04 主矩阵**数据落盘、结论暂缓**：在 cached_tokens/命中证据补齐前，README/简历不得引用任何「cache_aware 性能」数字。
- **补证（2026-08-30，worker metrics 差分）**：hot_prefix_1792 s1 发 20 个同前缀请求经 cache_aware router，w1 的 `prompt_tokens_total` delta = **43124**（≈20×2160 全量），w0 delta = 0。**结论**：cache_aware ① 把全部流量钉到 w1（负载 100/0）；② 且**没有命中前缀缓存**（命中则只记后缀，delta 应 ≪ 43200）。这坐实了高并发下 cache_aware TTFT 变差的双重根因：负载失衡（w0 空转 + w1 过载）叠加缓存未命中（前缀没进 radix tree 或命中判定失败）。**根因方向：cache_aware 在 8B 上的「近似树匹配」未命中 + 串行冷启动把前缀钉单卡（EXP-P06 同款），而非缓存命中后仍有固定开销。**
- 已把「router 响应丢 prompt_tokens_details」+「cache_aware 8B 上未命中前缀」登记为 S05 的第一优先排查项。
