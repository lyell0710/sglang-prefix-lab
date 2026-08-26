# EXP-P02 · token 契约矩阵(含一处预注册假设证伪)

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 环境 | venv sglang-lab · GPU0 · worker 28000(参数同 EXP-P01《env 与单 worker smoke》)· evidence sha 1378595 |
| 状态 | 完成 |
| 关联清单项 | docs/PLAN.md#exp-p02 |

## 1. 目的与假设(跑前锁定,预期写在 PLAN 与脚本 docstring)

五格双发矩阵,每格 flush 后 A/B 两发,读 B 的 cached_tokens:
base_messages ≈ n−1;thinking_flip **从 system 段分叉、hit ≪ base**;
input_ids_direct = n−1;salt_same ≈ n−1;salt_diff = 0。

## 2-3. 配置与步骤

`scripts/contract_matrix.py`(每格前 POST /flush_cache 并确认返回)。
排障两跳:①transformers 5.x `apply_chat_template(tokenize=True)` 返回
BatchEncoding 非 list(取 `["input_ids"]`);②首跑 preflight 因 /proc 竞态
在 set -e 下误崩(加固为段内 set +e)。

## 4. 原始数据

`data/raw/EXP-P02/20260824T163438_{preflight.txt,contract_matrix.json(+.prov),
engine_metrics.txt,template_divergence.json}`

## 5. 结果(B 发 cached/prompt)

| 格 | 结果 | 判定 |
|---|---|---|
| base_messages | 1324/1325 = n−1 | 符合预期 |
| **thinking_flip** | **1326/1329** | **预期证伪** |
| input_ids_direct | 1324/1325 | 符合预期 |
| salt_same | 1324/1325 | 符合预期 |
| salt_diff | absent(=0)| 符合预期(命名空间隔离)|

## 6. 分析与结论

- **证伪与改判**:CPU 侧渲染对比(raw=template_divergence.json)证明 Qwen3 模板的
  `enable_thinking=False` 是**纯尾部扩展**(1325 token 原样 + `<think>\n\n</think>\n\n`
  4 token,首分叉位=1325)——开关在 generation prompt 段,不在 system 段。
  因此 thinking 配置不一致**不破坏**前缀共享。theory/01/03 已按实测改写。
- **意外收获**:命中 1326 = 1325+1,多出的 1 token 是 A 请求的**首个输出 token**
  (`<think>`)——radix 树缓存 input+output 全序列(cache_finished_req)在接口层
  的直接可观测证据。
- cached_tokens 报告语义:0/冷启动时字段**缺失**而非 0(usage_processor 只在 >0
  时带 details)——下游脚本一律 `or 0` 处理。
- 五格中四格按预注册预期,一格证伪且归因到 file:line + raw,契约结论可用于
  P03+ 的 manifest 设计(正式臂用 input_ids 直传,salt 做人工 miss 对照)。

## 7. 异常、偏差与开放问题

- flush_cache 返回体是文本非 JSON("Cache flushed...")——脚本按前缀截断记录,
  成功判定靠后续 A 发 cached 缺失(冷)佐证;更严格的 success 布尔在
  /flush_cache?timeout= 的 JSON 路径,P03 起改用状态码+冷验证双确认。
- thinking_flip 的 B 命中含输出重叠 1 token,若换 system-prompt 开头非 `<think>`
  的模型该 +1 不复现——结论限定 Qwen3 模板族。

- 追记(08-24 审计):首跑残留 20260824T163333_*(含 0 字节 json,两次中断产物)已移 data/archive/EXP-P02/ 附原因。

## 8. 下游影响

- P03/P04 manifest:input_ids 直传为正式形态;messages 形态保留 probe 臂。
- theory/03 §2.2 与 theory/01 §5 已修订(错误推断不过夜——CORE 铁律 7)。
- 面试素材:「预注册→证伪→CPU 复核→file:line 归因」完整小案例。
