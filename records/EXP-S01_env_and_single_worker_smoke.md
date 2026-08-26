# EXP-S01 · 独立环境与单 worker smoke

> 迁移注（2026-08-25）：自 sglang-inference-lab（Codex 建仓）并入本仓，编号沿用；文内 scripts/ 指原仓工装（git 历史 582fc6a/3a091ad 可查），现行工装=本仓 scripts/。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 协议 | `config/protocol-v1.json`；本 EXP 不产性能结论 |
| 环境 | `sglang-lab` + SGLang `v0.5.18@71de97b264b0` |
| 状态 | 完成；一次并发撞车 FAIL 完整保留 |
| 关联清单项 | `docs/PLAN.md#exp-s01--environment-and-single-worker-smoke` |

## 1. 目的、假设与预锁阈值

建立不污染 vLLM 的独立 CUDA 13 环境。PASS：依赖检查无冲突、Torch 识别两张 sm89 GPU、单个 SGLang worker 在一张空闲 GPU 上完成启动，`/health`、`/v1/models`、`/v1/chat/completions`、`/metrics`、 `/model_info` 均返回 2xx，响应正文非空，进程能按 PID/PGID 清理。

## 2. 环境与配置

- venv：`/root/venvs/sglang-lab`，Python 3.12.11；完整 204-package freeze 见 raw。
- upstream：独立 detached worktree `/root/repos/sglang-v0.5.18`，SHA `71de97b264b04dcd514cf904003028aefe9775c8`。
- smoke：Qwen3-0.6B revision `c1899de...`，GPU1，TP=1，context 4096，BF16， `mem-fraction-static=0.35`，默认 FlashInfer、prefill/decode CUDA Graph 与 Radix cache。
- 服务名/端口：`smoke1` / 18001；模型名固定为 `Qwen/Qwen3-0.6B`。

## 3. 可复现步骤

```bash
uv venv --python 3.12 /root/venvs/sglang-lab
uv pip install --python /root/venvs/sglang-lab/bin/python 'sglang==0.5.18' 'sglang-router==0.3.2'

scripts/service_ctl.sh start-worker smoke1 1 18001 \
  /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  Qwen/Qwen3-0.6B 4096 0.35
scripts/wait_http.py http://127.0.0.1:18001/health --timeout 300
scripts/run_smoke.sh smoke1 http://127.0.0.1:18001 Qwen/Qwen3-0.6B EXP-S01
scripts/archive_runtime_log.sh EXP-S01 smoke1 mem035_single_worker PASS
scripts/service_ctl.sh stop smoke1
```

## 4. 原始数据

- `data/raw/EXP-S01/20260824T161031_env_summary.txt`、`*_pip_freeze.txt`、`*_dependency_check.txt`。
- `data/raw/EXP-S01/20260824T161818_smoke1_mem035_single_worker.log`：隔离后 PASS 的完整启动/请求日志。
- `data/raw/EXP-S01/20260824T161800_smoke1_{smoke_response,metrics,model_info,gpu_snapshot}.txt`。
- `data/raw/EXP-S01/20260824T161425_smoke0_default_graph_capture_oom.log` 与 `20260824T161851_w0_concurrent_collision_other_worker.log`：并发撞车失败的两侧日志。
- `data/raw/EXP-S01/20260824T161835_concurrent_worker_collision_correction.txt`：对前一文件误导性标签的修正。

## 5. 直接测量结果

| 检查 | 结果 |
|---|---|
| 包版本 | SGLang 0.5.18；router 0.3.2；Torch 2.13.0+cu130；FlashInfer 0.6.17 |
| 依赖完整性 | 204 packages checked，0 incompatibility |
| CUDA | 2 GPUs available；两张均 RTX 4090 capability 8.9 |
| worker 启动 | PASS；RadixCache 初始化，默认 prefill/decode graph 捕获完成 |
| KV pool | 64,403 tokens（本 smoke 配置） |
| 稳态显存快照 | GPU1 9,914 MiB；GPU0 1 MiB |
| OpenAI smoke | models=200、chat=200、正文非空、finish_reason=stop |
| 可观测性 | `/metrics`、`/model_info` 均 200，metrics 含 request/cache/TTFT/E2E 系列 |
| 清理 | 项目进程组退出后两张卡均回到 1 MiB / P8 |

这些数字只证明环境与工装可用，不是性能 benchmark。

## 6. 分析与结论

- **实测**：固定 v0.5.18 wheel 在本机 CUDA/driver/RTX 4090 上可正常 import、JIT、捕图、启动和服务请求； RadixCache 与 metrics codepath 都有日志/端点证据。
- **实测**：第一次 OOM 时有两个独立 agent 同时向 GPU0/port 18000 启动 worker；一侧错误明确列出另一 scheduler PID 629181 占 9.21 GiB，因此该 run 是协议碰撞，不能用于评价默认 CUDA Graph 内存。
- **结论**：EXP-S01 PASS；下一步先做不可变 workload 与逐 token reference，不报任何 latency 数字。

## 7. 异常、偏差与开放问题

- 首次 smoke payload 未关闭 Qwen3 thinking，16-token 上限截断思考文本；HTTP transport PASS，但不满足内容 gate。脚本随后固定 `chat_template_kwargs.enable_thinking=false`，隔离复跑正文非空且正常 stop。
- 并发 agent 创建了另一套 `worker_ctl.sh` 和第二个 worker，导致撞车。后续 GPU run 前必须同时检查 GPU process、端口和 `runtime/*.pid`，不能只相信自己的 PID 文件。
- `mem-fraction-static=0.35` 仅为 0.6B smoke 配置；8B 双副本配置必须在 EXP-S03 重新锁定。

## 8. 下游影响

- 解锁“已搭建固定版本的单 worker SGLang 实验环境”；不解锁双副本、cache-aware 或性能措辞。
- `service_ctl.sh` 采用精确 PID/进程组管理；失败 raw 不进入 derived。
- EXP-S02 需固定 Qwen3 thinking mode、tokenized 长度与 reference token IDs。

