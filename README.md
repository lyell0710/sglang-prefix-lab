# SGLang Prefix Lab · RadixAttention 机理与工作负载契约

在 2×RTX 4090 上把 SGLang v0.5.18 的**前缀缓存(RadixAttention)**从源码机制做到可复现测量与证伪:
每条量化主张都有 raw 数据、≥3 seeds、反例臂与引擎计数器闭环背书。
本仓证明的是**机制归因与证据链**,不是"把服务跑起来"。

> 本文件是唯一状态源(定位/EXP 台账/红线)。预注册协议 [docs/PLAN.md](docs/PLAN.md),
> 原理笔记 [docs/theory/](docs/theory/),过程记录 [LAB_JOURNAL.md](LAB_JOURNAL.md),
> 简历映射 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md)。

## 🎯 Headline(30 秒版)

| 结果 | 数字 | 证据指针 |
|---|---|---|
| 前缀命中收益(Qwen3-8B) | TTFT p50 228.4→52.9 ms(并发1,**−77%**)、1068.3→234.5 ms(并发8,**−78%**)@ 共享前缀 1792/2048;disable-radix 反例臂全线打平 | EXP-P07 · [data/derived/exp_p07_8b_ttft_vs_prefix.csv](data/derived/exp_p07_8b_ttft_vs_prefix.csv) |
| 逐 token 归因闭环 | 引擎计数器 device_hit = **466,944** 与客户端 Σcached_tokens 逐 token 相等(0.6B/8B 双复现) | EXP-P03/P07 §6 |
| LRU 逐出悬崖 | 池 < 重用距离 8192×(1+cr) 时命中 **1.0→0.0625 阶跃崩塌**,无中间态;三池×四档冷流量,seed 间 std=0 | EXP-P05 · [data/derived/exp_p05_eviction_cliff.csv](data/derived/exp_p05_eviction_cliff.csv) |
| 调度不是标量优劣(8B 积压档) | lpm p50 **−62%**、命中 **+17.7pp**,但 p99 **+64%**——延迟在分位数间再分配;0.6B 同协议(EXP-P04)则单纯反劣 | EXP-P08 · [data/derived/exp_p08_8b_fcfs_vs_lpm.csv](data/derived/exp_p08_8b_fcfs_vs_lpm.csv) |
| radix 命中首证 | 第二发 cached=**1324/1325(=n−1)**,精确命中源码"至少重算 1 token"的上限 | EXP-P01 · data/raw/EXP-P01/ |
| 预注册证伪 ×3(全程留痕) | Qwen3 thinking 开关=纯尾扩展不破坏共享(P02);rr 全命中=奇偶分片巧合、cache-aware 冷启动把热前缀全钉一卡(流量 100/0)而崩(P06 双证伪) | EXP-P02 / EXP-P06 |

## 📊 图表(全部由 scripts/ 从 data/ 生成)

![8B 命中收益曲线](figures/fig2_p07_ttft_vs_prefix_8b.png)

**8B 下前缀命中收益放大到 −77%/−78%,disable-radix 反例臂持平——收益确来自前缀复用,不是别的。**
src: [data/derived/exp_p07_8b_ttft_vs_prefix.csv](data/derived/exp_p07_8b_ttft_vs_prefix.csv)(EXP-P07,2026-08-24)· [scripts/plot_ttft_curve.py](scripts/plot_ttft_curve.py)

![0.6B 命中收益曲线](figures/fig1_p03_ttft_vs_prefix_0p6b.png)

**同协议 0.6B 只有 −36%/−63%:prefill 占比越小收益天花板越低——"命中收益正比于被跳过的 prefill"的直接量化。**
src: [data/derived/exp_p03_ttft_vs_prefix.csv](data/derived/exp_p03_ttft_vs_prefix.csv)(EXP-P03,2026-08-24)· [scripts/plot_ttft_curve.py](scripts/plot_ttft_curve.py)

![LRU 逐出悬崖](figures/fig3_p05_eviction_cliff.png)

**LRU 逐出不是斜坡是悬崖:池 ≥ 重用距离 D=8192×(1+cr) 则命中 1.0,越线即崩至 ~0.06,三池验证无一例外(std=0)——容量规划要按重用距离而不是热集大小。**
src: [data/derived/exp_p05_eviction_cliff.csv](data/derived/exp_p05_eviction_cliff.csv)(EXP-P05,2026-08-24)· [scripts/plot_eviction_cliff.py](scripts/plot_eviction_cliff.py)

![8B 调度权衡](figures/fig4_p08_sched_tradeoff.png)

**8B 积压档 lpm 把延迟从中位数搬到尾部换命中率(p50 −62%/hit +17.7pp/p99 +64%)——选 lpm 与否取决于 SLO 定义在 p50 还是 p99。**
src: [data/derived/exp_p08_8b_fcfs_vs_lpm.csv](data/derived/exp_p08_8b_fcfs_vs_lpm.csv)(EXP-P08,2026-08-24)· [scripts/plot_sched_tradeoff.py](scripts/plot_sched_tradeoff.py)

## 🔬 代码导览:TTFT 怎么停表,命中怎么闭环

节选自 [scripts/bench_prefix.py](scripts/bench_prefix.py)(`# ←` 为导读注释)。同一次流式请求里**既停表又拿到
server 侧命中数**,"快了多少"与"命中了多少"在同一响应内闭合,归因不靠猜:

```python
async def one_request(session, url, model, ids, out_tokens):
    payload = {"model": model, "input_ids": ids,        # ← token id 直传,绕过 chat template 渲染差异
               "messages": [{"role": "user", "content": "x"}],
               "temperature": 0.0, "max_tokens": out_tokens, "stream": True,
               "stream_options": {"include_usage": True}}  # ← 让 server 把 usage 随流返回
    t0 = time.perf_counter()
    ttft = None; cached = None
    async with session.post(url + "/v1/chat/completions", json=payload) as r:
        async for raw in r.content:
            ...
            ch = d.get("choices") or []
            if ttft is None and ch and (ch[0].get("delta") or {}).get("content"):
                ttft = (time.perf_counter() - t0) * 1e3  # ← 首个 content chunk 到达 = TTFT 停表
            u = d.get("usage")
            if u and u.get("prompt_tokens_details"):
                cached = u["prompt_tokens_details"].get("cached_tokens")  # ← server 报的命中 token 数
    return {"ttft_ms": ttft, "e2e_ms": e2e, "cached_tokens": cached}
```

机制一句话([docs/theory/01_radix_prefix_cache.md](docs/theory/01_radix_prefix_cache.md)):
服务端把每个请求的 **token id 序列**插进 radix tree(节点 value=KV 物理块索引),新请求最长前缀匹配后
直接复用 KV 跳过 prefill;调度侧把匹配上限压到 **input_len−1**——永远至少重算 1 个 token,否则没有
logits 可采样。实测锚:EXP-P01 第二发 cached=1324/1325,**正是 n−1 的精确值**,不是近似。

聚合侧的闭环校验在 [scripts/aggregate_p03.py](scripts/aggregate_p03.py):每个请求的 cached_tokens 必须
恰等于 prefix_len(ON 臂)或 0(OFF 臂),再与 /metrics 的 `device_hit` 计数器交叉对账。

## 🚀 复现 Quickstart

```bash
bash scripts/preflight.sh            # 两卡空闲 + 端口清白,否则中止
bash scripts/svc.sh start w0 0 28000 --model-path Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --tp 1 \
  --enable-metrics --enable-cache-report
bash scripts/svc.sh wait_health w0

# 命中收益曲线的一个 round(3 seeds = 换 --seed 重跑;raw 只写 UTC 前缀新文件)
/root/venvs/sglang-lab/bin/python scripts/bench_prefix.py \
  --base-url http://127.0.0.1:28000 --concurrency 8 --seed 20260824 \
  > data/raw/EXP-P03/$(date -u +%Y%m%dT%H%M)_radix_on_c8.jsonl

python scripts/aggregate_p03.py EXP-P03 data/derived/exp_p03_ttft_vs_prefix.csv
python scripts/plot_ttft_curve.py data/derived/exp_p03_ttft_vs_prefix.csv \
  figures/fig1_p03_ttft_vs_prefix_0p6b.png "标题=结论句" "Qwen3-0.6B"
bash scripts/svc.sh stop w0
```

其余协议同构:P05 用 [scripts/bench_evict.py](scripts/bench_evict.py) + [scripts/aggregate_evict.py](scripts/aggregate_evict.py),
P04/P08 用 [scripts/bench_groups.py](scripts/bench_groups.py) + [scripts/aggregate_groups.py](scripts/aggregate_groups.py),
P06 用 [scripts/bench_route_pool.py](scripts/bench_route_pool.py)。异地环境见 [ENV.md](ENV.md)。

## 🗂 仓库结构

```
├── README.md            # 唯一状态源:台账 / 关键数字 / 红线
├── records/             # EXP 八节记录(预注册假设 → 结果 → 证伪留痕)
├── data/raw/EXP-*/      # 原始测量,首行 provenance,不可变
├── data/derived/        # 聚合脚本重算的稳定性表(mean±std)
├── figures/             # fig1-4,全部脚本生成,禁手改
├── scripts/             # svc / preflight / bench_* / aggregate_* / plot_*
├── docs/theory/         # 源码机理笔记 01-03(file:line 锚)
├── docs/PLAN.md         # 预注册协议(判定阈值跑前锁定)
├── docs/talk/           # 面试讲稿
├── config/              # 协议 JSON(protocol-v1 / protocol-router-v1)
└── LAB_JOURNAL.md       # 顺写日记:做了什么/为什么/关键数字/产物
```

## 与相邻项目的边界

| 项目 | 分工 |
|---|---|
| ~~sglang-inference-lab~~(2026-08-25 并入本仓) | 其 EXP-S00/S01 与 router 矩阵协议已收编;完整历史在本仓 git;旧路径留指路牌 |
| `vllm/experiments` | vLLM 侧部署形态/PD/MoE;本仓只做 SGLang engine 侧前缀机理 |
| `llm-engine` / kernel 仓 | 算子与手写引擎;本仓不把服务 wall-clock 冒充 kernel 数字 |

资源纪律:与 sibling 共享 2×4090 与 venv;本仓端口 28000/28001/40000/29000,
**任何 GPU 实验前 [scripts/preflight.sh](scripts/preflight.sh) 必须通过**(外来 compute 进程/端口占用即中止)。

## 🧾 EXP 台账

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---:|:---:|---|
| EXP-S00 | bootstrap_audit(并入) | 2026-08-24 | ✅ | 60s 超时事故排查+host 体检;GPU/端口清白证明 → data/raw/EXP-S00/ |
| EXP-S01 | env_and_single_worker_smoke(并入) | 2026-08-24 | ✅ | venv sglang-lab 安装验证(现行共用环境的出生证明)→ records/EXP-S01 |
| EXP-P01 | env_single_worker_smoke | 2026-08-24 | ✅ | 确定性✓;第二发 cached=1324/1325(=n−1);hit_rate 0.9992;flashinfer 后端 → data/raw/EXP-P01/ |
| EXP-P02 | token_contract_matrix | 2026-08-24 | ✅ | 5 格:4 格符合预注册,thinking_flip 证伪(Qwen3 开关是纯尾扩展,命中 1326/1329)→ data/raw/EXP-P02/ |
| EXP-P03 | hit_benefit_curve | 2026-08-24 | ✅ | TTFT p50:c1 −36%/c8 −63%(prefix 1792/2048);device_hit 计数与 Σcached 逐 token 相等;OFF 臂平 → data/derived/exp_p03_ttft_vs_prefix.csv |
| EXP-P04 | lpm_vs_fcfs | 2026-08-24 | ✅ | std 档无可区分;boundary 档(192req>128 窗口)lpm p99 反劣 13%、hit −2.4pp(2σ)→ data/derived/exp_p04_fcfs_vs_lpm.csv |
| EXP-P05 | eviction_pressure | 2026-08-24 | ✅ | LRU 悬崖:池<重用距离(8192×(1+cr))时 hit 1.0→0.0625 阶跃,三池验证,std=0 → data/derived/exp_p05_eviction_cliff.csv |
| EXP-P07 | 8b_hit_benefit_curve | 2026-08-24 | ✅ | Qwen3-8B:TTFT p50 −77%(c1)/−78%(c8)@prefix 1792/2048;device_hit 逐 token 闭环复现;off 臂平 → data/derived/exp_p07_8b_ttft_vs_prefix.csv |
| EXP-P08 | 8b_scheduling_tradeoff | 2026-08-24 | ✅ | 8B boundary:lpm p50 −62%/hit +17.7pp 但 p99 +64%——分位数再分配,不是标量优劣 → data/derived/exp_p08_8b_fcfs_vs_lpm.csv |
| EXP-P06 | routing_pool_capacity | 2026-08-24 | ✅ | 双预测双证伪:rr@偶数热集=奇偶分片巧合全命中;cache_aware 冷启动全钉一卡而崩(100/0 流量);hot5 对照坐实 → data/derived/exp_p06_routing_pool.csv |

## 🧭 方法论与措辞红线(诚实度文化)

本仓把"诚实"做成机械流程而不是态度:**每个 raw 文件首行 provenance**(env/sha/完整命令/日期/GPU),
进 README 的关键数字一律 **≥3 独立 seeds** 带 mean±std;每条收益主张配**反例臂**(disable-radix)与
计数器闭环对账;预注册假设被证伪**不删记录**——P02/P05/P06/P08 的证伪与修正全程留痕在 records/,
勘误后的旧数字只存在于按时间序保留的史料(LAB_JOURNAL/records),现行文档禁止两代数字并存。
下表 gate 一切对外表述:

| 主张 | 状态 | 解锁条件 |
|---|---|---|
| "搭建 SGLang 前缀缓存实验台(单 worker)" | ✅ 已解锁(EXP-P01,08-24)| — |
| "前缀命中使 TTFT 降 77%/78%"(8B) | ✅ 已解锁 | EXP-P07 全 gate PASS(3 seeds+反例臂+计数器闭环)|
| "lpm 调度提升命中率/尾延迟"(无定语) | 🚫 永久禁用 | P04(0.6B 反劣)与 P08(8B 分位数再分配)证明须带模型/负载/分位数定语 |
| "router cache-aware 提升 TTFT/吞吐"类主张 | ⛔ | 本仓只测了容量受限机理格(P06);性能矩阵属 S02-S07 未执行 |
| "生产级/多机/集群" | 🚫 | 超出硬件与实验范围 |

## 🗺 路线图(未执行阶段)

双副本 router 性能矩阵(S02-S07,预注册协议=[docs/PLAN_router_matrix.md](docs/PLAN_router_matrix.md) +
[config/protocol-router-v1.json](config/protocol-router-v1.json))——serving 部署故事的下一阶段,需整机独占。

## 🔗 相关仓

- [github.com/lyell0710/vllmExperience](https://github.com/lyell0710/vllmExperience) — vLLM 侧部署形态/PD/MoE 证据仓
- [github.com/lyell0710/Kernel_Optimazation](https://github.com/lyell0710/Kernel_Optimazation) — CUDA kernel 优化证据仓
- [github.com/lyell0710/llm-engine](https://github.com/lyell0710/llm-engine) — 手写推理引擎
- 本仓暂无远端;外部背书状态见 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md) 缺口台账
