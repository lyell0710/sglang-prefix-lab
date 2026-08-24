# SGLang Prefix Lab · RadixAttention 机理与工作负载契约

> 本文件是唯一状态源(定位/边界/EXP 台账/红线)。协议见 [docs/PLAN.md](docs/PLAN.md),
> 原理笔记 docs/theory/01-03,过程记录 [LAB_JOURNAL.md](LAB_JOURNAL.md)。

## 定位

在 RTX 4090 上把 SGLang v0.5.18 的**前缀缓存(RadixAttention)**从源码机制做到可复现
测量:token 级契约 → 命中收益曲线 → 调度放大 → 逐出退化。主角是**机制归因与证据链**,
不是"把服务跑起来"。

## 与相邻项目的边界

| 项目 | 分工 |
|---|---|
| `/root/projects/sglang-inference-lab`(sibling agent 并行仓) | 双副本 **router 策略矩阵**(cache_aware vs round_robin);本仓不重复,仅在 EXP-P06 交叉复核 |
| `vllm/experiments` | vLLM 侧部署形态/PD/MoE;本仓只做 SGLang engine 侧前缀机理 |
| `llm-engine` / kernel 仓 | 算子与手写引擎;本仓不把服务 wall-clock 冒充 kernel 数字 |

资源纪律:与 sibling 共享 2×4090 与 venv;本仓端口 28000/28001/40000/29000,
**任何 GPU 实验前 `scripts/preflight.sh` 必须通过**(外来 compute 进程/端口占用即中止)。

## EXP 台账

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---:|:---:|---|
| EXP-P01 | env_single_worker_smoke | — | ⬜ | 无 |
| EXP-P02 | token_contract_matrix | — | ⬜ | 无 |
| EXP-P03 | hit_benefit_curve | — | ⬜ | 无 |
| EXP-P04 | lpm_vs_fcfs | — | ⬜ | 无 |
| EXP-P05 | eviction_pressure | — | ⬜ | 无 |
| EXP-P06 | dual_replica_crosscheck(扩展) | — | ⬜ | 无 |

## 当前关键数字

暂无(理论笔记与协议已就绪,未产生性能数字)。

## 措辞红线

| 主张 | 状态 | 解锁条件 |
|---|---|---|
| "搭建 SGLang 前缀缓存实验台" | ⛔ | EXP-P01 全 PASS |
| "前缀命中使 TTFT 降 X%" | ⛔ | EXP-P03:3 round + 反例臂(disable-radix)+ server 侧归因 |
| "lpm 调度提升命中率/尾延迟" | ⛔ | EXP-P04 gate 全过 |
| "router cache-aware ..."(任何路由主张) | 🚫 | 属 sibling 仓范围;仅 P06 交叉复核后可引用其编号 |
| "生产级/多机/集群" | 🚫 | 超出硬件与实验范围 |

## 怎么跑

```bash
bash scripts/preflight.sh            # 必须 exit 0
bash scripts/svc.sh start w0 0 28000 --model-path Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --tp 1 \
  --enable-metrics --enable-cache-report
bash scripts/svc.sh wait_health w0 && bash scripts/svc.sh stop w0
```
