# EXP-P06 · 路由 × 池容量:预注册预测被双向证伪,机理由对照钉死

> **一句话结论**：两条预注册假设**双向证伪**，各有干净机理：rr@hot6 全命中是轮转周期与 worker 数奇偶对齐造成的完美分片；cache-aware 亲和只有在 tenant 分散到多卡时才等效于扩池，低负载下它的冷启动分配反而集中，亲和与容量相乘为负。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 环境 | venv sglang-lab · 双卡（w0/w1 各 --max-total-tokens 8192）· router=sgl-model-gateway(sglang-router 0.3.2)@40000 · evidence sha 9590a56 |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN.md#exp-p06（**范围偏离**：原预注册为"与 sibling S04 交叉复核"，因 sibling 未产出 S04，依 PLAN 预留的 fallback 改为本仓框架内的机理实验，见 §7）|

## 1. 目的与假设(跑前锁定,由 EXP-P05 重用距离模型推出)

双 worker 各限池 8192，热工作集 6×~2150 > 单池 < 双池和。预测：
- H-rr：round_robin 把同一前缀交替打到两卡 → 每卡看到全部前缀 → 双卡 thrash，hit→~0
- H-ca：cache_aware 把每前缀钉在一张卡 → 每卡 ~3 前缀 → hit→~1（"路由=扩大有效池"）

## 2. 环境与配置

`scripts/bench_route_pool.py` v2：**文本形态**负载（随机 token 解码文本，这正是 cache_aware 近似树的匹配对象）；6(/5)个热前缀轮转 ×4，串行 c=1；逐请求 cached_tokens + prompt_tokens 硬 gate（<768 即 FAIL）；两 worker `prompt_tokens_total` 差分定位流量；每臂前置 gate：router prometheus 的 `policy` 标签必须等于本臂。3 seeds。

## 3. 步骤

preflight → 双 worker → 每策略：router 起 → policy 标签 gate → 3 seed bench → selection 指标抓取 → router 停 → hot5 奇偶对照重复全程。

## 4. 原始数据

`data/raw/EXP-P06/20260824T165910_*`（12 组 bench json + 4 组 router 指标 + preflight；16:53 首轮作废数据保留原地，作废原因见 §7）；聚合 `data/derived/exp_p06_routing_pool.csv`。

## 5. 结果(3 seeds,全部格 std=0;流量 split 来自 worker 计数器差分)

| 配置 | 策略 | hot_hit | 流量分布 |
|---|---|---|---|
| hot6（偶） | round_robin | **1.0000**（24/24 全命中）| ~50/50(30878/30921)|
| hot6（偶） | cache_aware | **0.0020**(0/24)| **100/0**(61799/0)|
| hot5（奇，对照） | round_robin | 0.0020(0/20)| ~50/50 |
| hot5（奇，对照） | cache_aware | 0.0020(0/20)| 100/0(51507/0)|

## 6. 分析与结论

- **两条预注册假设都错了，且各有干净的机理**（实测+对照）： ① rr@hot6 全命中：轮转周期 6 与 worker 数 2 **奇偶对齐**，严格轮询意外成为完美分片——每卡只见 3 个前缀（距离 ~6450 < 8192）。hot5 对照打破奇偶后 rr 立即崩塌（流量仍均分，但每卡要装全部前缀，~10750 > 8192）——机理坐实。 ② cache_aware 全崩：worker 计数器直接显示**全部流量落在一张卡**。串行 c=1 下负载恒 0，失衡回退（64/1.5 阈值）永不触发；冷启动 tenant 分配把 6 个前缀全钉到同卡，亲和性把 ~12900 的工作集塞进 8192 的单池 → thrash。
- **修正后的结论**：cache-aware 亲和只有在 tenant **分散**在多卡时才等效于扩大池容量；它的冷启动分配在低负载下会**集中**，此时亲和性与容量约束相乘为负。 rr 的"命中"则是脆弱的巧合（依赖轮转周期与卡数的整除关系），不可依赖。
- 一句话：**在容量受限的多副本里，前缀→副本的映射质量（分散且稳定）比 "cache-aware"这个标签重要；两种现成策略都不保证这一点**。

## 7. 异常、偏差与开放问题

- **范围偏离**：P06 预注册主体为交叉复核，sibling S04 未产出，按预留 fallback 改为机理实验；交叉复核在 sibling 完成后仍可补（编号沿用其台账）。
- **首轮（16:53）整批作废**，两个教训：① router 严格按 OpenAI schema 重序列化， **丢弃 `input_ids` 扩展字段**，全部请求静默退化为 ~10 token（靠 cached=8、 worker 增量 135 反推发现）——修正：文本负载 + prompt_tokens 硬 gate； ② router 用 setproctitle 改名 `sglang::router`，svc.sh 身份校验落空 → 第一支 router 存活跨臂，cache_aware 臂实际仍是 rr（selection 指标 policy 标签抓包坐实）——修正：身份放行三形态 + 每臂 policy 标签前置 gate。作废 raw 保留。
- 开放：cache_aware 冷启动全落一卡的内部路径未定（候选：min-load 平局的确定性选择 vs 文档所称随机 tie-break）。**追记（08-24）：两次日志级尝试均失败**—— ①RUST_LOG=debug 未透过 Python 包装生效（0 条 debug）；②`--log-level debug` 生效（170 条 debug）但 sglang-router 0.3.2 构建**不输出决策级日志**（全部为 job_queue/MCP/registry 基建行，无 select/tenant/match_rate 行；raw= data/raw/EXP-P06/20260824T175130_cache_aware_debug_rationale.txt）。结论维持 "内部路径未定"，确证需读 wheel 内实际源码或上游加日志——列为潜在 issue 素材。
- 开放：非严格轮转的真实到达序（Poisson/并发）下 rr 的奇偶巧合会被打破， cache_aware 的失衡回退会被激活——高并发行为未测，不外推。

## 8. 下游影响

- 解锁措辞：「用重用距离模型预注册双预测并**双向证伪**，以奇偶对照与流量计数器差分钉死两个机理：rr 的整除巧合分片、cache-aware 低负载亲和集中；结论=容量受限多副本下映射分散度决定成败（12 组 bench，seed 间 std=0）」。
- theory/02 需补一节「容量受限下的亲和集中」引用本记录。
- 若做上游贡献：cache_aware 冷启动 tenant 分配的分散性是可提 issue 的观察点（先完成开放问题 1 的 debug 日志复核）。
