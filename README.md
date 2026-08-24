# SGLang Inference Lab · 前缀缓存与双副本路由

> 本文件是项目的**唯一状态源**：项目边界、运行入口、EXP 台账、当前数字和简历措辞红线均以此为准。
> 实验协议见 [docs/PLAN.md](docs/PLAN.md)，学习路线见 [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md)，过程记录见
> [LAB_JOURNAL.md](LAB_JOURNAL.md)。

## 定位

在 2×RTX 4090 上搭建可复现的 SGLang serving 实验台，研究一个明确问题：

> 当请求具有可控的共享前缀时，`cache_aware` 路由相对 `round_robin` 在什么负载区间改善 TTFT；
> 什么时候又会因负载失衡而失去收益？

项目主角是**工作负载设计、请求级证据和边界归因**，不是“把服务跑起来”。所有 headline 数字必须由固定
workload manifest、正确性 gate、router/worker metrics、GPU telemetry 和至少 3 个独立 round 共同支撑。

### 与现有项目的边界

| 项目 | 本项目不重复的部分 |
|---|---|
| `vllm/experiments` | 已覆盖 PD/NIXL、TP、MoE 与量化；本项目只研究 SGLang RadixAttention、共享前缀和副本路由 |
| `llm-engine` | 已覆盖手写 dense forward/KV cache；本项目研究在线 serving 的请求分配与尾延迟 |
| `triton-kernels` / `Kernel_Optimazation` | 已覆盖算子；本项目不把服务级 wall clock 冒充 kernel benchmark |

边界：当前只做单机双 GPU、同模型双副本；不声称多机、NVLink/IB、Hopper、训练或生产部署经验。

## 固定基线

- SGLang：稳定版 `v0.5.18`（被测 SHA 在 EXP-S01 固化；开发中的 `main` 不作性能基线）。
- 模型：`Qwen/Qwen3-0.6B` 只做低成本 smoke；正式矩阵使用 `Qwen/Qwen3-8B`，revision
  `b968826d9c46dd6066d109eabc6255188de91218`。
- 拓扑：GPU0/GPU1 各一个 TP=1 worker，router 单独进程。
- 主对照：`round_robin` vs `cache_aware`；无共享前缀是必须保留的反例。
- 环境：独立 `/root/venvs/sglang-lab`；不得复用或污染 vLLM 环境。

## 怎么跑

当前处于 bootstrap，先运行无副作用体检：

```bash
cd /root/projects/sglang-inference-lab
bash scripts/preflight.sh
```

环境和服务入口会在 EXP-S01 smoke 通过后写入本节；在此之前不提供未经验证的“一键 benchmark”命令。
远端安装方法见 [ENV.md](ENV.md)，本机路径只登记在
`/root/work/infra/machine/ENV_REGISTRY.md`。

## EXP 索引台账

状态：⬜ 计划 / ⛔ 阻塞 / ◐ 进行中 / ✅ 完成。只有“记录 + raw + commit”齐全才可标 ✅；远程创建后再补 push gate。

| 编号 | slug | 日期 | 状态 | 关键数字（指针） |
|---|---|---:|:---:|---|
| EXP-S00 | bootstrap_audit | 2026-08-24 | ✅ | GPU/端口空闲；超时最可能来自 host I/O 阻塞（records/EXP-S00） |
| EXP-S01 | env_and_single_worker_smoke | — | ⬜ | 无 |
| EXP-S02 | correctness_and_workload_contract | — | ⬜ | 无 |
| EXP-S03 | dual_replica_router_observability | — | ⬜ | 无 |
| EXP-S04 | routing_policy_matrix | — | ⬜ | 无 |
| EXP-S05 | boundary_and_profile_attribution | — | ⬜ | 无 |
| EXP-S06 | repeatability_and_resume_evidence | — | ⬜ | 无 |
| EXP-S07 | upstream_gap_and_pr_gate | — | ⬜ | 无 |

## 当前关键数字

暂无可用于简历的性能数字。环境体检结果是运维事实，不是性能结论。

## 措辞红线

| 主张 | 当前状态 | 解锁条件 / 证据 |
|---|---|---|
| “搭建 SGLang 双副本服务实验台” | ⛔ | EXP-S03：两个 worker、router、health/metrics 和正确性全部 PASS |
| “cache-aware 使 TTFT 改善 X%” | ⛔ | EXP-S04：同 manifest 配对 A/B、≥3 rounds、mean±std、全 gates PASS |
| “定位收益边界/负载失衡回退” | ⛔ | EXP-S05：命中路径证据 + router/worker metrics + profile-only 归因 |
| “完成上游贡献/PR” | ⛔ | 实际提交后才可写“submitted”；合并后才可写“merged” |
| “生产级/集群级/多机” | 🚫 | 不在本项目硬件和实验范围内 |

## 证据规则

- raw 永不覆盖；失败运行保留并写 `gates.pass=false`，协议错误数据移入 `data/archive/`。
- derived/figure 只能读取明确 manifest 中 gate PASS 的 run，禁止“自动取 latest”。
- profiler run 标记 `PROFILE_ONLY`，其时延不进入性能表。
- SGLang SHA 指被测 upstream 版本，不是本证据仓 SHA。
- 没有 raw 的数字只能称“终端级证据”，不得进入 README headline 或简历。
