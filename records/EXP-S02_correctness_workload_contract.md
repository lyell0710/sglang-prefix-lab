# EXP-S02 · correctness and workload contract（router 矩阵前置）

> **一句话结论**：router 矩阵的不可变 manifest（3 workload × 3 seeds × 192 请求）与单 worker 确定性 reference（9 × 8 probe）落盘，SHA256 可复现——S04 主矩阵的「同一 manifest」「parity probe」两条 gate 的地基。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | venv sglang-lab · GPU0 · Qwen3-8B(bf16, revision b968826…) @ 28000 · mem-fraction 0.85 |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN_router_matrix.md EXP-S02 |

## 1. 目的与假设

按预注册协议生成不可变 manifest 并采单 worker 确定性 reference。PASS 判定：manifest 的 SHA256 跨 A/B 两臂相同（同输入）；probe 集在 temperature=0 下与单 worker reference 逐 token 一致（worker 行为不被 router 改变）。

## 2. 环境与配置

- 单 worker：`svc.sh start w0 0 28000 --model-path …/Qwen3-8B/snapshots/b968826… --served-model-name Qwen/Qwen3-8B --mem-fraction-static 0.85 --context-length 4096 --tp-size 1`。
- 工装：`scripts/gen_router_manifest.py`（生成）、`scripts/gen_router_reference.py`（采集）。

## 3. 步骤

1. 生成 manifest（3 workload × 3 seeds × 192 请求，token 确定生成）。
2. 启动单 worker，健康后采 reference（每 manifest 前 8 请求，temperature=0）。

## 4. 原始数据

- `data/raw/EXP-S02/{unique_control,hot_prefix_1024,hot_prefix_1792}_s{2026082401..03}.jsonl`（9 文件，SHA256 见 §5）。
- `data/raw/EXP-S02/reference.json`（9 × 8 probe 的确定性输出 + SHA256）。

## 5. 结果

manifest SHA256（节选，全量见 reference.json summary）：

| workload | seed | sha256 前缀 | req0 重编码 token |
|---|---|---|---|
| unique_control | s1 | 78ae422c… | 2139 |
| hot_prefix_1024 | s1 | b26a8399… | 2152 |
| hot_prefix_1792 | s1 | 1ea9956c… | 2148 |

（3 seeds × 3 workload 全量 9 份，n=192；重编码 token 2139–2156，比名义 2048 多 6–8%（decode→encode 漂移，bench_route_pool 同法，以响应 prompt_tokens 为准）。

## 6. 分析与结论

- **manifest 不可变**：token 由固定 seed 确定性生成（`random.choices(seed)`），decode 成文本落盘 JSONL，SHA256 可复现——S04 A/B 两臂读同一 manifest 的前提。
- **reference 确定**：temperature=0 + max_tokens=32，串行采集，SHA256 可复现。
- **一个协议过时登记**：protocol-router-v1.json 的 `worker_ports=[18000,18001]`、`router_port=30000` 是原仓 sglang-inference-lab 的旧值；本仓工装（svc.sh/preflight.sh）用 **28000/28001/40000**。实际以本仓工装为准，protocol 字段不改（预注册文本按史料保留，偏离在记录登记）。

## 7. 异常、偏差与开放问题

- **thinking 修正（本记录重要勘误）**：初版 reference 未关 Qwen3 的 enable_thinking，thinking 输出即使 temperature=0 也非确定（位置 78 起分叉）——EXP-S01 §7 已知此坑，S02 初版遗漏。已重采：`chat_template_kwargs={"enable_thinking": False}` 后 parity 逐 token 一致。S04 的 bench 必须同关。
- 重编码 token 漂移（2048→2139~2156）：`tok.decode(ids)` 再 encode 不精确往返（tokenizer 的 BPE 合并边界），bench_route_pool 的硬 gate 下限取 `prefix_len//2` 正是为此——本 manifest 的 gate 同样以响应 prompt_tokens 为准，不追求精确等长。
- 8B 加载 + CUDA graph capture 185s（冷启动），S03/S04 的 worker 生命周期管理要预留。
- worker metrics 默认关（v0.5.18 `enable_metrics=False`），需显式 `--enable-metrics` 否则 /metrics 404——S03 已用此参数重启 worker。

## 8. 下游影响

- 解锁 S04 的「同一 manifest SHA256」「parity probe 逐 token 一致」两条 gate。
- S03 直接用本 manifest + reference 做双 worker + router 的可观测性验证。
