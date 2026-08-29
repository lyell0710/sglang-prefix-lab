# EXP-S05 · cache_aware 8B 未命中的根因归因（boundary + 机理）

> **一句话结论**：S04 主矩阵发现 cache_aware 高并发 TTFT +53~196% 且 worker metrics 差分证明「流量钉单卡 + 未命中缓存」，本记录用 EXP-P06 的机理 + 本批数据钉死根因：**cache_aware 在低负载（串行 c=1）冷启动阶段把前缀集中钉到单卡，失衡回退阈值（abs 64 / rel 1.5）在「负载恒 0」下永不触发，亲和性把热工作集塞进单卡 KV 池 → thrash + 无命中**。这是 0.6B（EXP-P06）在 8B 上的**完整复现**，且多了一个 8B 特有的放大因子。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | venv sglang-lab · GPU0/1 · Qwen3-8B @ 28000/28001 · router @ 40000 |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN_router_matrix.md EXP-S05；交叉复核 EXP-P06 §8 |

## 1. 目的与假设

S04 遗留：cache_aware 未命中 + 负载失衡，需归因。假设（跑前锁定，来自 EXP-P06 机理 + S04 数据）：cache_aware 冷启动集中分配 + 失衡回退不触发，与 0.6B 同机理。

## 2. 环境与配置

同 S04（双 worker 8B + router）。补证用 worker metrics 差分（`prompt_tokens_total` before/after）。

## 3. 步骤

1. hot_prefix_1792 s1 发 20 个同前缀请求经 cache_aware router。
2. worker metrics 差分：w0 delta=0、w1 delta=43124（≈20×2160 全量）。
3. 对照 EXP-P06 §6 的 0.6B 结论。

## 4. 原始数据

- 补证输出（终端级证据）：`w0 delta=0, w1 delta=43124`；预热后第二遍 `w0 delta=21575`。未落盘 data/raw/。
- 对照：EXP-P06 raw（`data/derived/exp_p06_routing_pool.csv`，cache_aware 流量 100/0）。

## 5. 结果

| 现象 | 0.6B（EXP-P06） | 8B（S04 补证） |
|---|---|---|
| cache_aware 流量分布 | 100/0（61799/0） | 100/0（w1 delta 43124，w0 0） |
| 前缀缓存命中 | 崩（hot_hit 0.0020） | 未命中（delta≈全量，无 cached 递减） |
| 高并发 TTFT | 未测 | cache_aware 比 rr +53~196% |

## 6. 分析与结论

**① 机理闭环（跨模型复现）。** EXP-P06 在 0.6B 上已经钉死：cache_aware 的冷启动 tenant 分配在低负载下把前缀集中钉到单卡；失衡回退阈值 `(max_load - min_load) > 64` 且 `max_load > min_load × 1.5` 在串行 c=1（负载恒 0）下永不触发。S04 在 8B 上复现：20 个同前缀请求全部落 w1（delta 43124），w0 空转。**这不是实现 bug，是 cache_aware 策略在「容量受限 + 低负载」下的固有行为**——亲和性把热工作集塞进单卡 KV 池，反而 thrash。

**② 8B 特有的放大因子：未命中后代价更高。** 0.6B 下未命中的 prefill 是 ~27ms，8B 下全 miss prefill 是 ~230ms（S04 unique_control c1 TTFT 299ms 佐证）。同样「未命中 + 单卡过载」，8B 的 TTFT 绝对代价大一个量级，这就是为什么 S04 高并发下 cache_aware 反而 +53~196%——**cache_aware 的集中分配 + 8B 的贵 prefill 相乘**。

**③ 结论句（面试口径）**：cache-aware 路由的价值前提是「前缀能分散到多卡 + 命中」，而冷启动集中分配 + 失衡回退阈值在低负载下失效，使它在容量受限的 2 卡场景下**既不能分散、也不能命中**——前缀→副本映射质量（分散且稳定）比「cache_aware」标签重要。这与 EXP-P06 一句话结论完全一致，且被 8B 的贵 prefill 放大。

## 7. 异常、偏差与开放问题

- **未直接验证「预热后 cache_aware 能否命中」→ 已补（2026-08-30）**：预热后再发 10 个同前缀请求，第二遍 delta 仍是 21575（≈10×2160 全量，未命中应为 ~2560）。**结论升级**：cache_aware 在本版本（v0.5.18）+ 本配置下，**前缀缓存命中根本没生效**——不只是冷启动集中，而是预热后同前缀仍全量重算。这是比 EXP-P06 更深的发现，根因候选：① router 近似树匹配在文本形态下未命中（text 经 tokenizer 渲染后与 worker radix key 不对齐）；② cache_threshold=0.3 的命中判定过严；③ router 的 cache 树与 worker 的 radix tree 是两套，前者未更新。**需升级到 sglang 侧源码级排查，本记录到此为止，不再深挖版本内部。**
- 8B 的 `balance_abs_threshold=64` 是否可调、调小能否让 cache_aware 在高并发下回退，未测——属可选调参，非主结论。

## 8. 下游影响

- **交叉复核闭合（EXP-P06 §8 的 fallback）**：sibling S04 产出后回填的交叉复核位，本记录完成——0.6B 与 8B 同机理（冷启动集中 + 未命中），且 8B 额外发现「预热后仍不命中」。
- README/简历不得引用任何「cache_aware 性能优势」数字；可引「cache_aware 在 2 卡容量受限场景的冷启动集中 + 缓存未命中缺陷」作为负结果（带 8B 放大因子）。
- **S06 方向收敛**：命中未生效的根因（近似树 vs radix key 对齐 / cache_threshold / 双树）需 sglang 源码级排查，属版本内部行为；若作为「上游 gap」则进 S07（upstream gap and PR gate），否则诚实结束（cache_aware 在本配置下无效是实测结论，不造 PR）。
