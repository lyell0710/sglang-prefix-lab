---
topic: SGLang bench_serving(v0.5.18)用于共享前缀路由实验的关键事实
status: 源码级核对完成(只读),file:line 以 /root/repos/sglang-v0.5.18 为准
source: python/sglang/benchmark/serving.py(bench_serving.py 只是 shim,:1-22)
---

# bench_serving · 为 EXP-S02/S04 设计 workload 前必须知道的事

## 1. generated-shared-prefix(GSP)数据集

- 类 `datasets/generated_shared_prefix.py:39-120`，生成器 `:154-328`。
- 参数（`serving.py:2610-2699`）:`--gsp-num-groups`(64)`--gsp-prompts-per-group`(16) `--gsp-system-prompt-len`(2048)`--gsp-question-len`(128)`--gsp-output-len`(256) `--gsp-range-ratio`(1.0)`--gsp-num-turns`(1)`--gsp-ordered` `--gsp-group-distribution uniform|zipf` `--gsp-zipf-alpha` `--gsp-send-routing-key`。
- **prompt = 随机 token 解码成的文本**(`datasets/common.py:132-136`)，不是自然语言； 拼接 `f"{system_prompt}\n\n{question}"`(`:284`)。→ 我们自建 manifest 时要用 token 级长度校验（S02 gate），不能信字符长度。
- **请求总数 = num_groups × prompts_per_group,`--num-prompts` 对 GSP 无效**(`:86-100`)。
- **默认 shuffle**(`:300-301`)，`--gsp-ordered` 才按组连续发。
- 磁盘缓存 `~/.cache/sglang/benchmark/gen_shared_prefix_*.pkl`(`:123-151`)； `range_ratio!=1 / send_routing_key / num_turns!=1` 时不缓存（`:197`）。
- routing key：每组一个 header `X-SMG-Routing-Key`(`serving.py:54`)，只在 `--gsp-send-routing-key` 时发。

## 2. 调度旋钮

- `--request-rate`（默认 inf=全部 t=0 发出；否则 Poisson，`:2366-2372`，`:1053-1088`）。
- `--max-concurrency` → asyncio.Semaphore(`:1379-1386`)。**按列表顺序派发**(`:1516-1554`)， `outputs[i]` 与 `input_requests[i]` 对齐（`:1556`）。
- `--seed`(42，`:2445`)在数据集加载前生效（`:1973-1974`）→ shuffle 可复现。
- `--warmup-requests`(1,`:2598-2603`)，用 `input_requests[0]`,output≤32(`:1424-1433`)。
- backend:`sglang`(/generate)`sglang-oai`(/v1/completions)`sglang-oai-chat`(/v1/chat/completions) (`:939-972`)。`--output-file`(JSONL)`--output-details`（逐请求数组）`--tag`。

## 3. 指标定义(报告里每个数字的出处)

- TTFT = 首个含文本的流式 chunk 到达时刻 − 请求发出 `st`(`:747-749`)； E2E = 最后一个 SSE chunk − `st`(`:767`)；TPOT = (E2E−TTFT)/(output_len−1)(`:1116-1117`)； ITL = 相邻 chunk 间隔，多 token chunk 均摊（`:755-760`）。
- 汇总 mean/median/std/p90/p95/p99(`:1233-1262`)；JSONL 字段表 `:1789-1841`（**不含 gsp_* 参数** → 必须 `--tag` 区分）。
- `--output-details` 才写逐请求 `ttfts/itls/input_lens/output_lens/errors`(`:1875-1896`)； **无逐请求 e2e 数组**，只能 ttft+Σitl 重建（与 latency 定义是否严格相等：UNVERIFIED）。

## 4. 缓存命中与冲刷

- `--cache-report`(`:2439-2444`)：sglang 原生路径读 `meta_info.cached_tokens`(`:736-741`)； OAI 路径注入 `return_cached_tokens_details`(`:1982-1988`)，读 `sglext.cached_tokens_details` (`:244-254`)。汇总命中率打印 `:1725-1779`。
- `--flush-cache`(`:2587-2591`)：warmup 后、正式计时前 `POST /flush_cache`(`:1454-1461`，`:977-995`)。 → 每个 A/B arm 的冷启动定义 = flush 后开始。

## 5. chat vs completions,tokenizer

- `sglang-oai-chat` 把 prompt 包成单条 user message(`:427`);**GSP 不套 chat template** (`generated_shared_prefix.py:57`)。多轮 wrapper 只对 *-chat backend(`:1297-1334`)。
- tokenizer 来源：`--tokenizer` → `/model_info` → `--model`(`:2091-2103`)。

## 6. benchmark/ 目录里可借鉴的

- `benchmark/hicache/bench_warm_cache.py`：**共享前缀比例研究**（flush → 预热前缀 → 随机后缀， `--pcts/--total-tokens/--num-prompts/--max-concurrency`，`:557-637`）——与本项目 S02 最近。
- `benchmark/hicache/bench_multiturn.py`：多轮会话负载生成器。
- **仓内没有 router 专用 benchmark**（grep 验证）——本项目的双副本路由 bench 是自建工装。

## 对本项目的设计含义

1. 正式矩阵可直接用 GSP + `--gsp-*` 生成负载，但 **manifest 固化与 token 长度校验仍由我们自己做** (S02 gate)；`--gsp-ordered` 与默认 shuffle 分别对应"友好/对抗"两种到达序。
2. 每个 arm:`--flush-cache --cache-report --output-details --tag <arm> --output-file <raw>`。
3. 热组偏斜用 `--gsp-group-distribution zipf`，这是 S05 边界 sweep 的现成旋钮。
