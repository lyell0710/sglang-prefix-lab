# EXP 记录模板（复制本文件为 EXP-Dxx_<slug>.md，当场写）

## 项目级附则（对数值方法与阈值 · 跑之前锁定，跑完不许改）

- **对数值统一方法**：同 SEED、同输入；自研引擎逐层 dump 中间 tensor，与 HF `output_hidden_states=True` / 各层 forward hook 输出对齐；记 **max_abs_err + mean_abs_err**。
- **PASS 阈值**：max_abs_err < 1e-2（BF16）/ < 1e-3（FP32）。生成类：贪心逐 token 完全一致。
- **参考实现版本写死**：transformers / torch 版本以 ENV.md（llm-engine#EXP-D01 固化）为准，记录中只写指针。
- **SEED**：每个测量点唯一 SEED，写进配置表；性能数字 ≥3 轮，mean/std 落 stability 文件。
- **大 dump 铁律**：入库「首末层 + 抽样中间层 + 最终 logits top-k」；全量本体放 data/local/（gitignore），仓里入 manifest（路径 + sha256 + provenance 行）。
- raw 首行 provenance：`# provenance: env= sha= cmd= date= gpu= driver= seed=`。

---

# EXP-Dxx · <实验标题>

| 字段 | 值 |
|---|---|
| 日期 | YYYY-MM-DD |
| 环境 | ENV（venv 名 + 本仓 commit sha） |
| 状态 | 完成 / 部分完成 / 作废（写明原因） |
| 关联清单项 | docs/PLAN.md#EXP-Dxx |

## 1. 目的与假设
一句话目的；可证伪假设 + 判定阈值（引用头部附则或写更严的本实验阈值）。

## 2. 环境与配置
模型 + 尺寸 + dtype + seq_len + SEED + commit；完整命令与环境变量；硬件占用（哪张卡）；共享状态（权重缓存冷热、是否复用进程）。

## 3. 步骤
可复现的命令序列（或指向脚本 + 参数）。

## 4. 原始数据
文件指针列表（data/raw/EXP-Dxx/...，每个文件首行 provenance）；任何只来自终端输出而无 raw 的数字**必须在此注明**「终端级证据」。

## 5. 结果
关键数字表（max_abs_err / mean_abs_err，或 TTFT / TPOT / 吞吐，单位进表头）。只放实测，不放推论。

## 6. 分析与结论
与假设对照；实测与推断分开标注。一句话结论 + 下一步。

## 7. 异常、偏差与开放问题
协议偏离（改了参数要写）、未解释现象；开放问题写去向（哪个 EXP 跟进）。

## 8. 下游影响
对 README 红线表 / 工装 / theory 笔记 / 简历句的影响。
