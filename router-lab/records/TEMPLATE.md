# EXP-Sxx · <实验标题>

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | YYYY-MM-DD |
| 协议 | `config/protocol-v1.json` 或明确的新版本 |
| 环境 | `sglang-lab` + 被测 SGLang SHA/tag |
| 状态 | 完成 / 部分完成 / 作废（写原因） |
| 关联清单项 | `docs/PLAN.md#...` |

## 1. 目的、假设与预锁阈值

写一个可证伪假设；列出开始跑之前就确定的 PASS/FAIL 和实用显著性阈值。

## 2. 环境与配置

模型 revision、dtype、worker/router 参数、GPU 映射、workload manifest SHA、seed、进程生命周期、缓存冷热状态。

## 3. 可复现步骤

完整命令或脚本 + 参数；秘密信息不得落盘。说明停止/清理方法。

## 4. 原始数据

列出 `data/raw/EXP-Sxx/` 指针。文本第一行必须有 provenance；JSONL/二进制必须有 sidecar/manifest。
没有 raw 的内容明确标注“终端级证据”。

## 5. 直接测量结果

只放实测：请求数/失败、TTFT/TPOT/E2E、吞吐、metrics、GPU 状态、正确性结果；单位进入表头。

## 6. 分析与结论

把“实测”与“推断”分开。逐条对照假设，给出一句带边界的结论。

## 7. 异常、偏差与开放问题

记录失败 run、协议偏离、silent fallback、计时扰动、无法解释的现象和后续 EXP 去向。

## 8. 下游影响

说明 README 红线、代码/工装、学习笔记、上游候选和简历句是否变化。

