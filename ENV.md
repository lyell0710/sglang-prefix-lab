# ENV · 异地复现

本机实况见 `/root/work/infra/machine/ENV_REGISTRY.md`(venv `sglang-lab`)。本文件只讲异地复现。

- Linux,Python 3.12,NVIDIA sm ≥ 8.0（本机 2×RTX 4090 / sm89,driver 610.57.04,CUDA 13.2）。
- 固定 SGLang `v0.5.18`。安装：
  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python 'sglang==0.5.18' sglang-router==0.3.2
  ```
  实测栈（freeze 见 records/data/*_freeze.txt）:torch 2.13.0+cu13、flashinfer 0.6.17、 sglang-kernel 0.4.6.post1、triton 3.7.1、transformers 5.12.1。
- 模型（固定 revision）：
  - `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`(smoke)
  - `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`（正式矩阵/正确性参考）
