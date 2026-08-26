# EXP-P01 · env 与单 worker smoke(radix 首证)

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 环境 | venv sglang-lab(freeze=records/data/20260824T161135_sglang_lab_freeze.txt)· GPU0 · evidence sha 1378595 |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN.md#exp-p01 |

## 1. 目的与假设(跑前锁定)

单 worker 起服 + 确定性 + radix 活性首证。判定:①/health、/v1/models 200;
②同 payload 双发 temperature=0 输出逐字相同;③第二发
`usage.prompt_tokens_details.cached_tokens > 0` 且 ≤ prompt_tokens−1。

## 2. 环境与配置

`svc.sh start w0 0 28000 --model-path Qwen/Qwen3-0.6B --revision c1899de… --tp 1
--mem-fraction-static 0.8 --enable-metrics --enable-cache-report`;preflight 通过
(双卡空闲、端口空闲、无外来 compute 进程)。共享状态:与 sibling 仓共用 venv 与
GPU,本次窗口其 worker 未运行。

## 3. 步骤

见 README「怎么跑」+ `scripts/probe_cached.py`(探针:长 user 文本 ~1325 token,
双发对比)。

## 4. 原始数据

`data/raw/EXP-P01/20260824T162947_{preflight.txt,probe_cached.json(+.prov),
gpu_after.csv,engine_metrics.txt,w0_startup.log}`

## 5. 结果

- health 35s;attention backend=**flashinfer**(默认选择,与 theory 预测一致);
  Load weight avail mem=**23.07 GB**。
- probe:models_ok ✓ / deterministic ✓ / **第二发 cached_tokens=1324,
  prompt_tokens=1325** → 命中 = prompt−1,精确等于"至少重算 1 token"上限。
- engine /metrics:`sglang:cache_hit_rate=0.9992`、
  `cached_tokens_total{device}=1324`。

## 6. 分析与结论

三判定全 PASS。cached=n−1 不是近似而是精确值(page_size=1 + 上限
input_len−1,theory/01 §2.2)——机制笔记的第一处实测锚。
**并回溯闭环一个悬案**:16:12 首次尝试(在 sibling 仓)avail mem 仅 9.17GB 并
CUDA graph OOM,本次同卡 23.07GB → 当时是与 sibling agent 的 worker 同卡相撞
(其 16:21 提交的 EXP-S01《独立环境与单 worker smoke》同窗口在跑),不是驱动/容器限制。

## 7. 异常、偏差与开放问题

- `svc.sh stop` 首次拒停:uv venv 的 python 是指向 miniconda 真身的符号链接,
  /proc/exe 解析后与 `$VENV/bin/python` 字面不等 → 身份校验误判。已修
  (readlink -f 双侧归一);修后 pgid TERM 正常。
- pgid TERM 后一个子进程(640244)慢 ~2s 退出,自行消失;stop 增加的 30s 等待
  已覆盖该窗口,无遗留。
- 首次(16:12)的 OOM 现场原始日志留在 sibling 仓工作树时被我清理退出,该段仅
  余本会话终端输出 → **终端级证据**(结论不受影响,9.17 vs 23.07 的对照有 raw)。

## 8. 下游影响

- 解锁 P02-P05;README 台账 P01 ✅。
- 措辞可解锁:「搭建 SGLang 前缀缓存实验台(单 worker)」;双 worker 表述仍 ⛔。
- 运维规矩追加:任何 GPU 实验前 preflight 强制(外来 compute 进程即中止)——
  本次相撞的制度化答案。
