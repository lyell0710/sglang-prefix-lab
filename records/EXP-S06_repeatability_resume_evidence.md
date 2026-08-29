# EXP-S06 · repeatability and resume evidence（可复现性 + 简历草案）

> **一句话结论**：S02-S05 全流程从 manifest 可复现（SHA256 钉死的输入 + 确定性 reference），矩阵 54 cell 可重放；简历草案形成——但**结论是负结果**：cache_aware 在 2×4090 容量受限场景下「既不能分散、也不能命中」，这是本矩阵最有价值的产出，按「负结果 + 机理」入简历，不造「优化」句。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | venv sglang-lab · GPU0/1 · Qwen3-8B · router |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN_router_matrix.md EXP-S06 |

## 1. 目的与假设

验证全流程可复现性，形成带硬件/模型/负载/证据边界的简历草案。PASS 判定：manifest 重放产生一致结果；简历句每个数字可指到 gate 通过的证据。

## 2. 环境与配置

同 S02-S05。manifest（`data/raw/EXP-S02/`，SHA256 可复现）+ reference + 54 cell bench。

## 3. 步骤

1. 核对 manifest SHA256 与 reference 一致性（S02 已做）。
2. 核对矩阵 54 cell 的 3 seeds std（aggregate 已算）。
3. 形成简历草案。

## 4. 原始数据

- manifest 9 文件 + reference.json（SHA256 见 S02）。
- `data/raw/EXP-S04/derived_matrix.csv`（18 cell × 3 seeds）。
- 3 seeds 一致性：S04 各 cell 的 3 seeds p50 差异极小（c1 各 workload 两策略差 <3%）。

## 5. 结果

可复现性：manifest 确定（固定 seed token 生成），reference 确定（temperature=0 + 关 thinking），54 cell 各 3 seeds 聚合 std 小（TTFT p50 跨 seed 差异 <5%）。全流程可重放。

## 6. 分析与结论

**简历草案（负结果句，带完整边界）**：

> 在 2×RTX 4090 上构建 SGLang 双副本 router 实验台（v0.5.18，Qwen3-8B，2 policy × 3 负载 × 3 并发 × 3 seeds = 54 cell 预注册矩阵），测得 cache_aware 路由在高并发下 TTFT 反而 +53~196%，并用 worker 计数器差分钉死根因：**冷启动把前缀集中钉单卡（流量 100/0）+ 失衡回退阈值在低负载下失效 + 预热后同前缀仍不命中缓存**——前缀→副本映射质量（分散且稳定）比「cache_aware」标签重要，0.6B 与 8B 跨模型复现。

**证据边界**：①「cache_aware 命中未生效」是 v0.5.18 + 本配置的实测结论，未深挖版本内部根因（属 S07 上游 gap 范畴）；②TTFT 百分比是「cache_aware vs round_robin 同协议自比」，round_robin 在 hot_prefix 下的均衡是奇偶巧合（EXP-P06），不是「rr 更优」的普适结论；③ 结论只适用于「2 卡容量受限 + 串行冷启动」场景。

## 7. 异常、偏差与开放问题

- 吞吐字段 `throughput_req_s` 在 c=1 公式有误（S04 §7 已登记），S06 简历草案不用吞吐、只用 TTFT/TPOT，规避。
- 简历草案的「−77%」（P07）与「+53~196%」（S04/S05）是**不同实验不同结论**，不可混在一句里——前者是单 worker 前缀缓存命中收益，后者是双 worker cache_aware 路由缺陷。

## 8. 下游影响

- RESUME_EVIDENCE 主句更新：把 S04/S05 的「cache_aware 路由缺陷（8B，54 cell）」补进 S1 或 S2 弹药，交叉复核位闭环。
- S07（upstream gap）：cache_aware 命中未生效是否上游 bug，查 issue/PR；无真实缺口则诚实结束。
