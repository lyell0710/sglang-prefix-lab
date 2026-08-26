# EXP-P03 · 命中收益曲线:TTFT vs 共享前缀长度(radix on/off 双臂)

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 环境 | venv sglang-lab · GPU0 · Qwen3-0.6B @ 28000 · evidence sha fda054a |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN.md#exp-p03 |

## 1. 目的与假设(跑前锁定)

TTFT 随共享前缀长度近线性下降,降幅来源于 prefill 段;`--disable-radix-cache`
反例臂全线打平。判定:①ON 臂 cached 逐请求 = prefix_len;②OFF 臂 TTFT 对
prefix_len 无趋势;③ON 臂收益与 engine 侧 prefill 命中计数一致。

## 2. 环境与配置

`scripts/bench_prefix.py`:input_ids 直传(EXP-P02《token 契约矩阵》契约结论),total=2048 token,
prefix ∈ {0,512,1024,1536,1792},每点 16 请求,输出 32,流式首 chunk 停表;
每点先 /flush_cache,预热 1 条前缀请求(计时外);seed×3(20260824/25/26);
ON 臂并发 {1,8},OFF 臂并发 1。

## 3. 步骤

preflight → ON 臂(6 组 jsonl)→ 抓 /metrics → stop → OFF 臂(`--disable-radix-cache`,
3 组)→ stop → `scripts/aggregate_p03.py` 出 derived。

## 4. 原始数据

`data/raw/EXP-P03/20260824T163812_*`(9 个 jsonl 各带 provenance 首行 + preflight
+ radix_on_metrics.txt);聚合 `data/derived/exp_p03_ttft_vs_prefix.csv`。

## 5. 结果(TTFT p50,3 seed mean±std,ms)

| prefix/2048 | ON c1 | ON c8 | OFF c1 |
|---|---|---|---|
| 0 | 26.84±0.03 | 115.14±0.30 | 26.87±0.05 |
| 512 | 24.15±0.03 | 103.18±0.24 | 26.80±0.07 |
| 1024 | 19.37±0.09 | 73.31±0.18 | 26.68±0.18 |
| 1536 | 17.42±0.11 | 53.02±0.42 | 26.61±0.08 |
| 1792 | **17.27±0.08(−36%)** | **42.73±0.13(−63%)** | 26.59±0.15(平)|

cached 校验:ON 臂每请求 cached **精确 = prefix_len**(cached_ok 全 True);
OFF 臂与 pl=0 全部无 cached。

图:`figures/fig1_p03_ttft_vs_prefix_0p6b.png`(plot_ttft_curve.py 生成,2026-08-24)。

## 6. 分析与结论

- **三判定全成立**。反例臂钉死因果:收益只在 radix 开启且确有共享前缀时出现。
- **token 级归因闭环(实测)**:engine `prefill_effective_tokens_total{device_hit}
  =466,944`,与客户端 Σcached = 3 seed × 2 并发 × Σ(16×prefix_len) = 466,944
  **逐 token 相等**——TTFT 降幅确实且仅由省掉的 prefill token 兑现。
- **并发放大(实测→机理推断)**:c1 收益 −36%,c8 达 −63%。c1 下 TTFT 含 ~17ms
  与 prefill 无关的地板(0.6B 模型小,launch/采样/流式开销占比大);c8 下 prefill
  算力成为队列瓶颈,省掉的 prefill 同时缩短本请求与队友的排队,收益复利。
  斜率:c1 ≈ 5.3µs/token,c8 ≈ 40µs/token(排队放大系数 ~7.6)。
- `cache_hit_rate` 终值 0.0 的坑:窗口化 gauge(按 log 间隔重算),空闲后归零——
  **不能**当累计命中率读;累计口径用 `prefill_effective_tokens_total` 两条 counter。

## 7. 异常、偏差与开放问题

- c8 的 mean < p50(如 104.9 vs 115.1):并发批内先完成者拉低均值,分布左偏,
  属预期;报告以 p50/p95 为主。
- 0.6B 的 17ms 地板压缩了相对收益;8B 模型 prefill 更重,预期相对收益更大——
  留作扩展点(不阻塞,当前结论限定 0.6B)。
- OFF 臂 pl=0→1792 有 0.28ms 微降趋势(26.87→26.59),量级 < 轮间波动 ×2,按
  协议写"无可区分差异"。

## 8. 下游影响

- README 关键数字更新;解锁措辞:「共享前缀 1792/2048 时 TTFT p50 −36%(并发 1)
  /−63%(并发 8),cached 逐请求精确核对,engine 计数器逐 token 闭环,含
  disable-radix 反例臂,3 seeds」。
- P04/P05 可直接复用 bench 工装与归因方法(counter 差分)。
