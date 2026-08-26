# LEDGER · 状态与措辞账本(对内)

> **本文件是状态与措辞的唯一权威；README 为对外版，措辞以本表为准。** 过程日记：[LAB_JOURNAL.md](LAB_JOURNAL.md) · 简历映射：[RESUME_EVIDENCE.md](RESUME_EVIDENCE.md) · 预注册协议：[docs/PLAN.md](docs/PLAN.md) / [docs/PLAN_router_matrix.md](docs/PLAN_router_matrix.md)

## 🧾 EXP 台账

| 编号 | 名称 | slug | 日期 | 状态 | 关键数字（指针） |
|---|---|---|---:|:---:|---|
| EXP-S00 | bootstrap 现场审计 | bootstrap_audit（并入） | 2026-08-24 | ✅ | 60s 超时事故排查+host 体检；GPU/端口清白证明 → data/raw/EXP-S00/ |
| EXP-S01 | 独立环境与单 worker smoke | env_and_single_worker_smoke（并入） | 2026-08-24 | ✅ | venv sglang-lab 安装验证（现行共用环境的出生证明）→ records/EXP-S01 |
| EXP-P01 | env 与单 worker smoke（radix 首证） | env_single_worker_smoke | 2026-08-24 | ✅ | 确定性✓；第二发 cached=1324/1325(=n−1);hit_rate 0.9992;flashinfer 后端 → data/raw/EXP-P01/ |
| EXP-P02 | token 契约矩阵（含一处预注册假设证伪） | token_contract_matrix | 2026-08-24 | ✅ | 5 格：4 格符合预注册，thinking_flip 证伪（Qwen3 开关是纯尾扩展，命中 1326/1329）→ data/raw/EXP-P02/ |
| EXP-P03 | 命中收益曲线：TTFT vs 共享前缀长度（radix on/off 双臂） | hit_benefit_curve | 2026-08-24 | ✅ | TTFT p50：c1 −36%/c8 −63%(prefix 1792/2048)；device_hit 计数与 Σcached 逐 token 相等；OFF 臂平 → data/derived/exp_p03_ttft_vs_prefix.csv |
| EXP-P04 | 调度策略：lpm vs fcfs（标准档无可区分；边界档 lpm 反劣） | lpm_vs_fcfs | 2026-08-24 | ✅ | std 档无可区分；boundary 档（192req>128 窗口）lpm p99 反劣 13%、hit −2.4pp(2σ)→ data/derived/exp_p04_fcfs_vs_lpm.csv |
| EXP-P05 | 逐出压力：LRU 下命中不是衰减，是重用距离越线即崩塌 | eviction_pressure | 2026-08-24 | ✅ | LRU 悬崖：池<重用距离(8192×(1+cr))时 hit 1.0→0.0625 阶跃，三池验证，std=0 → data/derived/exp_p05_eviction_cliff.csv |
| EXP-P07 | 8B 收益曲线：0.6B 结论在部署级模型上放大并复现 | 8b_hit_benefit_curve | 2026-08-24 | ✅ | Qwen3-8B：TTFT p50 −77%(c1)/−78%(c8)@prefix 1792/2048；device_hit 逐 token 闭环复现；off 臂平 → data/derived/exp_p07_8b_ttft_vs_prefix.csv |
| EXP-P08 | 8B 调度：lpm vs fcfs 从"谁更好"变成"分位数再分配" | 8b_scheduling_tradeoff | 2026-08-24 | ✅ | 8B boundary：lpm p50 −62%/hit +17.7pp 但 p99 +64%——分位数再分配，不是标量优劣 → data/derived/exp_p08_8b_fcfs_vs_lpm.csv |
| EXP-P06 | 路由 × 池容量：预注册预测被双向证伪，机理由对照钉死 | routing_pool_capacity | 2026-08-24 | ✅ | 双预测双证伪：rr@偶数热集=奇偶分片巧合全命中；cache_aware 冷启动全钉一卡而崩（100/0 流量）；hot5 对照坐实 → data/derived/exp_p06_routing_pool.csv |

## 🧭 方法论与措辞红线(诚实度文化)

本仓把"诚实"做成机械流程而不是态度：**每个 raw 文件首行 provenance**（env/sha/完整命令/日期/GPU）， 进对外文档的关键数字一律 **≥3 独立 seeds** 带 mean±std；每条收益主张配**反例臂**(disable-radix)与计数器闭环对账；预注册假设被证伪**不删记录**——P02/P05/P06/P08 的证伪与修正全程留痕在 records/， 勘误后的旧数字只存在于按时间序保留的史料（LAB_JOURNAL/records），现行文档禁止两代数字并存。下表 gate 一切对外表述（README/简历/讲稿措辞以本表为准）：

| 主张 | 状态 | 解锁条件 |
|---|---|---|
| "搭建 SGLang 前缀缓存实验台（单 worker）" | ✅ 已解锁（EXP-P01,08-24）|— |
| "前缀命中使 TTFT 降 77%/78%"(8B) | ✅ 已解锁 | EXP-P07 全 gate PASS（3 seeds+反例臂+计数器闭环）|
| "lpm 调度提升命中率/尾延迟"（无定语） | 🚫 永久禁用 | P04（0.6B 反劣）与 P08（8B 分位数再分配）证明须带模型/负载/分位数定语 |
| "router cache-aware 提升 TTFT/吞吐"类主张 | ⛔ | 本仓只测了容量受限机理格（P06）；性能矩阵属 S02-S07 未执行 |
| "生产级/多机/集群" | 🚫 | 超出硬件与实验范围 |

## 与相邻项目的边界

| 项目 | 分工 |
|---|---|
| ~~sglang-inference-lab~~（2026-08-25 并入本仓） | 其 EXP-S00/S01 与 router 矩阵协议已收编；完整历史在本仓 git；旧路径留指路牌 |
| `vllm/experiments` | vLLM 侧部署形态/PD/MoE；本仓只做 SGLang engine 侧前缀机理 |
| `llm-engine` / kernel 仓 | 算子与手写引擎；本仓不把服务 wall-clock 冒充 kernel 数字 |

资源纪律：与 sibling 共享 2×4090 与 venv；本仓端口 28000/28001/40000/29000， **任何 GPU 实验前 [scripts/preflight.sh](scripts/preflight.sh) 必须通过**（外来 compute 进程/端口占用即中止）。

## 🗺 路线图(未执行)与待办

- 双副本 router 性能矩阵（S02-S07，预注册协议=[docs/PLAN_router_matrix.md](docs/PLAN_router_matrix.md) + [config/protocol-router-v1.json](config/protocol-router-v1.json)）——serving 部署故事的下一阶段，**需整机独占，择机执行**。
- 远端：https://github.com/lyell0710/sglang-prefix-lab（2026-08-26 建仓并公开）。
- P06 交叉复核位：router 矩阵执行后回填（EXP-P06 §8 已留 fallback 注记）。
