# 实验计划 · RadixAttention 机理与工作负载契约(engine 侧)

> 协议 v1,2026-08-24 预注册。参数改动须新开协议版本并在 EXP 记录解释，不得覆盖本文件。与 sibling 仓（router 策略矩阵）的分工见 LEDGER.md 边界节。〔注 2026-08-25：该仓已并入本仓，其协议=docs/PLAN_router_matrix.md；本行为指针更新，协议参数未动〕

## 研究问题

RadixAttention 的前缀命中在**真实 serving 接口**下由什么决定、值多少毫秒、何时失效？ 具体拆成：①token 级契约（什么样的请求才共享前缀）；②命中收益曲线（TTFT 降幅 vs 共享前缀长度）；③调度放大（lpm vs fcfs）；④逐出退化（KV 池吃紧时命中率衰减）。

## 固定变量

- SGLang v0.5.18,venv `/root/venvs/sglang-lab`;Qwen3-0.6B(smoke)/ Qwen3-8B（正式）, revision 见 ENV.md;BF16,greedy(temperature=0)，输出固定 32 token。
- 单 worker,GPU0，端口 28000;`--enable-metrics --enable-cache-report`。
- 每个性能点 ≥3 独立 round（独立 seed）；每臂开始前 `/flush_cache` 且确认 success。
- manifest JSONL 固化（内容+顺序+sha256），正式臂用 `input_ids` 直传（theory/03 §2）。

## EXP 清单

### EXP-P01 · env 与单 worker smoke(解锁一切)
health / `/v1/models` / 确定性 probe；**同 prompt 双发，第二次 `usage.prompt_tokens_details.cached_tokens > 0`** = radix 活着的首证。记录启动日志（attention backend=flashinfer 预期）、显存、freeze。

### EXP-P02 · token 契约矩阵(功能实验,GPU 轻)
同一段文本以 {messages 包装， input_ids 直传} × {thinking 默认/关} × {同/异 cache_salt} 组合双发，记录每格 cached_tokens。预 registered 预期：input_ids+同 salt 命中 ≈ prompt-1； messages 包装命中但短（模板头折扣）；异 salt 全 miss；thinking 开关不一致 → 从 system 段分叉。任何一格与预期不符 = 发现，单独归因。

### EXP-P03 · 命中收益曲线(headline 候选)
固定总长 2Ki token，共享前缀 ∈ {0, 512, 1Ki， 1.5Ki， 1.75Ki}，预热前缀后测 TTFT（bench_serving 或自研 client，并发 1 与 8 两档）。预期 TTFT 随共享长度近线性下降， 斜率 ≈ prefill 每 token 成本；用 server 侧 `prefill_finished_time-forward_entry_time` 交叉验证降幅确实落在 prefill 段。反例臂：`--disable-radix-cache` 全线打平。

### EXP-P04 · 调度策略放大(lpm vs fcfs)
`--schedule-policy {fcfs,lpm}` × 共享前缀负载（GSP 数据集，组数×每组=192），并发 16。机制：lpm 按最长前缀匹配排序聚簇同前缀请求（schedule_policy.py：373-384），提高在池命中；fcfs 打散。测 TTFT p50/p99 与 `sglang:cache_hit_rate`。注意 lpm 在等待队列 >128 时退化 fcfs(：290-294)——并发档位设计避开该退化区，另设一档故意跨过它作边界证据。

### EXP-P05 · 逐出压力与命中退化
把 `--mem-fraction-static` 压低造小 KV 池，固定热前缀集轮询注入冷流量，观测 `sglang:evicted_tokens_total` 上升与 cache_hit_rate/TTFT 退化曲线；LRU 下热前缀何时被冲掉（与 `--radix-eviction-policy lfu` 对照一格，作方向性证据）。

### EXP-P06 ·(扩展,需整机空闲)双副本路由交叉复核
与 sibling 仓 S04 同 manifest 跑我方独立实现的采集链，数字交叉核对；不重复首创。 sibling 未完成则此项按其协议执行并注明。

## Gate(跑前锁定)

- 正确性：probe 集 temperature=0 逐 token 与首轮一致；失败请求=0。
- 契约：manifest sha256 两臂相同；cached_tokens 预期按 page 对齐与 n-1 折扣建模（theory/03 §4）。
- 测量：warmup 在计时外；flush success=true 才开臂；GPU 无外来 compute 进程（preflight 强制）；配对臂温差 ≤5°C；每点 ≥3 round 报 mean±std + p50/p95/p99。
- 主张：相对变化 <5% 或与轮间波动同量级 → 写"无可区分差异"；正向结果必须带反例臂。
