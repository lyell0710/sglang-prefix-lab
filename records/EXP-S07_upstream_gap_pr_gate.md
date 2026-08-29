# EXP-S07 · upstream gap and PR gate（cache_aware 缺陷的上游对应）

> **一句话结论**：S04/S05 独立测到的「cache_aware 在共享前缀场景既不能分散也不能命中」是**上游已承认的缺陷**，不是我们的实现错误——sglang 官方 RFC #34513 明确写「cache_aware degrades precisely on the shared-prefix case」，并给出机理（router 自记近似树、从不观察 worker 实际缓存/逐出）；本记录完成查重，**诚实结束，不为 PR 而造 PR**（上游已有 agent_aware 替代方案在途）。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | 无 GPU（查证） |
| 状态 | 完成（诚实结束） |
| 关联清单项 | docs/PLAN_router_matrix.md EXP-S07 |

## 1. 目的与假设

S05 遗留：cache_aware 命中未生效是否上游 bug。假设：上游已知此缺陷。

## 2. 环境与配置

GitHub issue/PR 检索（cache_aware 命中失效 + 共享前缀 + 近似树）。

## 3. 步骤

检索关键词：`sglang cache_aware not hitting`、`cache_aware shared prefix`、读 cache_aware.rs 源码。

## 4. 原始数据

- sglang RFC #34513「Agent-aware session affinity without routing keys」
- sglang PR #27430「use full conversation for PD chat cache-aware routing」（#26263）
- `sgl-model-gateway/src/policies/cache_aware.rs` 源码

## 5. 结果

| 来源 | 关键结论 |
|---|---|
| RFC #34513 | 「`cache_aware` degrades precisely on the shared-prefix case. On a 29K-shared-prefix workload it completed only 58/110 sessions while a sticky policy completed 110/110.」 |
| cache_aware.rs | router 维护自己的近似树（`tree.insert(text, worker_url)` 只在匹配成功后），**从不观察 worker 实际缓存/逐出**；首请求树空 → match_rate=0 → 走「最小负载」路径 → 树不记 → 后续同前缀请求继续 miss |
| PR #27430 | PD 模式下 chat 只取第一条 message 做路由文本（多轮丢失），已修复 |

## 6. 分析与结论

**① 我们的实测与上游承认的缺陷精确对齐。** S04/S05 独立测到「冷启动集中 + 预热后仍不命中」，RFC #34513 给出同一机理的官方表述。根因在 cache_aware.rs 的近似树设计：它记录的是**路由器自己的路由决策**，而非 worker 的实际 radix tree 状态——这是一个「路由器的 cache 模型与 worker 的真实 cache 状态脱节」的结构性缺陷，不是参数调优能解决的。

**② 结论：诚实结束，不造 PR。** 上游已：① 在 RFC #34513 明确承认缺陷；② 提出 agent_aware 替代方案（strips shared prefix + hashes distinguishing part，免疫 clustering）；③ 修了 PD 多轮路由文本（#27430）。本仓再提 PR 无增量价值，符合「没有真实缺口时允许诚实结束，不为 PR 而造 PR」的预注册边界。

**③ 但我们的数据仍有独立价值**：RFC #34513 用的是 29K 前缀的 agentic-trace，我们是 **2 卡容量受限 + 短前缀（1024/1792）+ 8B 贵 prefill** 的独立场景，证明缺陷在「短前缀 + 容量受限」下更早触发（前缀只有 1024 就集中钉单卡）。这是对上游缺陷的独立复现 + 边界扩展。

## 7. 异常、偏差与开放问题

- 未跑 agent_aware 对照（本版本 v0.5.18 无 agent_aware，该策略在新版本在途）。
- 本记录的「诚实结束」意味着 router 矩阵 S02-S07 全线收官，但结论是**负结果主导**：cache_aware 在本配置下无效，round_robin 的均衡是奇偶巧合，两者都不可依赖。

## 8. 下游影响

- router 矩阵（S02-S07）收官：S02 manifest/reference、S03 可观测性、S04 主矩阵、S05 根因、S06 可复现+简历、S07 上游查证。**总结论：2×4090 容量受限场景下，双副本 router 的 cache_aware 策略不提供前缀缓存收益，且集中分配导致性能劣化——负结果，机理闭环，上游印证。**
- RESUME_EVIDENCE 的「双副本 serving 性能矩阵」缺口已闭环（负结果 + 上游印证）。
- LEDGER 待办「sglang router 矩阵 S02-S07」可销账。
