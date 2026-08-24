# ENV · 异地复现

> 本文件只描述异地复现。本机 venv、模型缓存、端口和启动状态统一登记在
> `/root/work/infra/machine/ENV_REGISTRY.md`，两边互链但不复制动态状态。

## 支持范围

- Linux，Python ≥3.10。
- NVIDIA GPU compute capability ≥8.0；本项目实测目标为 2×RTX 4090（sm89）。
- CUDA 13 wheel 栈；本机 driver/toolkit 版本在 EXP-S01 raw 中固化。

## 固定版本

性能基线锁定 SGLang `v0.5.18`，不跟随 mutable `main` 或 `latest`。该版本官方依赖包括
Torch 2.13.0、FlashInfer 0.6.17、`sglang-kernel` 0.4.6.post1；最终完整 freeze 由 EXP-S01 生成。

建议复现方式：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow 'sglang==0.5.18' sglang-router
```

安装后必须运行 EXP-S01 smoke；“pip 成功”不等于 CUDA backend 可用。

官方依据：

- SGLang v0.5.18 release: https://github.com/sgl-project/sglang/releases/tag/v0.5.18
- Installation: https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx
- Quick start: https://github.com/sgl-project/sglang/blob/v0.5.18/docs/docs/get-started/quickstart.mdx
- Router guide: https://github.com/sgl-project/sglang/blob/v0.5.18/docs/docs/advanced_features/dp_dpa_smg_guide.mdx

## 模型

| 模型 | revision | 用途 |
|---|---|---|
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 低成本环境 smoke |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 正确性 reference 与正式双副本矩阵 |

异地下载时必须固定 revision；本地 snapshot 路径不得写进对外复现命令。

## 环境隔离

- SGLang venv 不能复用 `/root/venvs/main`：后者属于 vLLM/llm-engine，且 FlashInfer 版本不同。
- upstream 源码、实验证据和 runtime PID/log 三者分开：`/root/repos`、本仓、`runtime/`。
- `runtime/` 与 `data/local/` 不入 Git；需要留证的日志复制到对应 `data/raw/EXP-Sxx/` 并带 provenance。
